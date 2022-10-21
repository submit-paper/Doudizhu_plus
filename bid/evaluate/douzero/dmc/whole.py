import os
import threading
import time
import timeit
import pprint
import logging
import sys
import traceback
from collections import deque
import typing
from logging import handlers
import sys
import traceback
import numpy as np
from collections import Counter
import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model, Coach
from .arguments import parser
from .env_utils import Environment
from douzero.env import Env

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}


Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}



class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str) 
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh) 
        self.logger.addHandler(th)

log = Logger('all.log',level='info')
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def create_env(flags):
    return Env(flags.objective)

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_optimizers(flags, learner_model):
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha)
        optimizers[position] = optimizer
    return optimizers


def create_buffers(flags):
    T = flags.unroll_length
    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = []
    for device in range(torch.cuda.device_count()):
        buffers.append({})
        for position in positions:
            x_dim = 319 if position == 'landlord' else 430
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                obs_x_no_action=dict(size=(T, x_dim), dtype=torch.int8),
                obs_action=dict(size=(T, 54), dtype=torch.int8),
                obs_z=dict(size=(T, 5, 162), dtype=torch.int8),
            )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers

def create_coach_buffers(flags):
    T_coach = flags.coach_length
    buffers = []
    for device in range(torch.cuda.device_count()):
        buffers.append({})
        specs = dict(
            init_landlord=dict(size=(T_coach, 20), dtype=torch.int32),
            init_landlord_up=dict(size=(T_coach, 17), dtype=torch.int32),
            init_landlord_down=dict(size=(T_coach, 17), dtype=torch.int32),
            win_res=dict(size=(T_coach,1),dtype=torch.float32),
            win_pred=dict(size=(T_coach,1),dtype=torch.float32),
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                _buffers[key].append(_buffer)
        buffers[device] = _buffers
    return buffers

def act(i, device, free_queue, full_queue, coach_free_queue, coach_full_queue, model, coach_model, buffers, coach_buffers, flags, coach_thresh):
    positions = ['landlord', 'landlord_up', 'landlord_down']
    print(coach_thresh.value)
    try:
        T = flags.unroll_length
        T_coach = flags.coach_length
        log.logger.info('Device %i Actor %i started.', device, i)

        env = create_env(flags)
        
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        init_landlord_buf = []
        init_landlord_down_buf = []
        init_landlord_up_buf = []
        win_buf = []
        win_pred_buf = []
        size_coach = 0
        
        position, obs, env_output = env.initial()
        
        while True:
            init_landlord = obs['init_landlord'].unsqueeze(0)
            init_landlord_up = obs['init_landlord_up'].unsqueeze(0)
            init_landlord_down = obs['init_landlord_down'].unsqueeze(0)
            with coach_thresh.get_lock():
                gate = coach_thresh.value
            with torch.no_grad():
                pred_res = coach_model(init_landlord, init_landlord_down, init_landlord_up)
            pred_win = torch.sigmoid(pred_res)
            del init_landlord
            del init_landlord_down
            del init_landlord_up
            if pred_win < gate or pred_win > 1 - gate:
                position, obs, env_output = env.initial()
            else:
                init_landlord_buf.append(obs['init_landlord'])
                init_landlord_down_buf.append(obs['init_landlord_down'])
                init_landlord_up_buf.append(obs['init_landlord_up'])
                win_pred_buf.append(pred_win.squeeze(0))
                print(pred_win, gate)
                while True:
                    obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                    obs_z_buf[position].append(env_output['obs_z'])
                    with torch.no_grad():
                        agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
                    _action_idx = int(agent_output['action'].cpu().detach().numpy())
                    action = obs['legal_actions'][_action_idx]
                    obs_action_buf[position].append(_cards2tensor(action))
                    position, obs, env_output = env.step(action)
                    size[position] += 1
                    if env_output['done']:
                        if env_output['episode_return'] > 0:
                            win_buf.append(torch.tensor(1.).unsqueeze(0))
                        else:
                            win_buf.append(torch.tensor(0.).unsqueeze(0))
                        size_coach += 1
                        for p in positions:
                            diff = size[p] - len(target_buf[p])
                            if diff > 0:
                                done_buf[p].extend([False for _ in range(diff-1)])
                                done_buf[p].append(True)

                                episode_return = env_output['episode_return'] if p == 'landlord' else -env_output['episode_return']
                                episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                                episode_return_buf[p].append(episode_return)
                                target_buf[p].extend([episode_return for _ in range(diff)])
                        break
            if size_coach > T_coach:
                index = coach_free_queue.get()
                if index is None:
                    pass
                else:
                    for t in range(T_coach):
                        coach_buffers['init_landlord'][index][t, ...] = init_landlord_buf[t]
                        coach_buffers['init_landlord_down'][index][t, ...] = init_landlord_down_buf[t]
                        coach_buffers['init_landlord_up'][index][t, ...] = init_landlord_up_buf[t]
                        coach_buffers['win_res'][index][t, ...] = win_buf[t]
                        coach_buffers['win_pred'][index][t, ...] = win_pred_buf[t]
                    coach_full_queue.put(index)
                    init_landlord_buf = init_landlord_buf[T_coach:]
                    init_landlord_down_buf = init_landlord_down_buf[T_coach:]
                    init_landlord_up_buf = init_landlord_up_buf[T_coach:]
                    win_buf = win_buf[T_coach:]
                    win_pred_buf = win_pred_buf[T_coach:]
                    size_coach -= T_coach
            for p in positions:
                if size[p] > T: 
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = obs_x_no_action_buf[p][t]
                        buffers[p]['obs_action'][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        exc_type, exc_value, exc_obj = sys.exc_info()
        log.logger.error(traceback.format_exc())
        traceback.print_exc()
        print()
        raise e

def _cards2tensor(list_cards):
    if len(list_cards) == 0:
        return torch.zeros(54, dtype=torch.int8)

    matrix = np.zeros([4, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    matrix = np.concatenate((matrix.flatten('F'), jokers))
    matrix = torch.from_numpy(matrix)
    return matrix


class ExceptionThread(threading.Thread):

    def __init__(self, target=None, name=None, args=()):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args
        self._exc = None

    def run(self):
        try:
            if self._target:
                self._target(*self._args)
        except BaseException as e:
            self._exc = sys.exc_info()
            exc_type, exc_value, exc_obj = sys.exc_info()
            log.logger.error(traceback.format_exc())
            traceback.print_exc()
        finally:
            #Avoid a refcycle if the thread is running a function with
            #an argument that has a member that points to the thread.
            del self._target, self._args

    def join(self):
        threading.Thread.join(self)
        if self._exc:
            log.logger.info("Thread '%s' threw an exception: %s" % (self.getName(), self._exc[1]))

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss


def compute_acc(pred, label):
    right = 0
    n = len(label)
    for i in range(n):
        if pred[i][0] < 0.5:
            if label[i][0] == 0:
                right += 1
        elif pred[i][0] > 0.5:
            if label[i][0] == 1:
                right += 1
    return right / n

def learn_coach(coach_actor_models, model, batch, optimizer, criterion, flags, lock):
    device = torch.device('cuda:'+str(flags.training_device))  
    landlord = torch.flatten(batch['init_landlord'].to(device), 0, 1)
    landlord_up = torch.flatten(batch['init_landlord_up'].to(device), 0, 1)
    landlord_down = torch.flatten(batch['init_landlord_down'].to(device), 0, 1)
    win = torch.flatten(batch['win_res'].to(device), 0, 1)
    pred_win = torch.flatten(batch['win_pred'].to(device), 0, 1)
    pred = torch.mean(pred_win).item()
    pred_std = torch.std(pred_win).item()
    with lock:
        res = model(landlord, landlord_down, landlord_up)
        possi = torch.sigmoid(res)
        loss = criterion(res, win)
        accuracy = compute_acc(possi, win)
        stats = {
                'coach_loss': loss.item(),
                'pred_win': pred,
                'pred_win_std': pred_std,
                'pred_acc': accuracy,
            }
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()
        for coach_model in coach_actor_models:
                coach_model.load_state_dict(model.state_dict())

        return stats

def learn(position,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock):
    """Performs a learning (optimization) step."""
    device = torch.device('cuda:'+str(flags.training_device))   # 100*32*--
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
        
    with lock:
        learner_outputs = model(obs_z, obs_x, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+position: loss.item(),
        }
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        optimizer.step()

        for actor_model in actor_models:
            actor_model.get_model(position).load_state_dict(model.state_dict())
        return stats

def train_whole(flags):  
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size
    coach_thresh = mp.Value('f', 0)
    
    
    # Initialize actor models
    models = []
    coach_actor_models = []
    assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'
    for device in range(flags.num_actor_devices):
        model = Model(device=device)
        model.share_memory()
        model.eval()
        models.append(model)
        coa_model = Coach().to(torch.device('cuda:'+str(device)))
        coa_model.share_memory()
        coa_model.eval()
        coach_actor_models.append(coa_model)

    # Initialize buffers
    buffers = create_buffers(flags)
    coach_buffers = create_coach_buffers(flags)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = []
    full_queue = []
    coach_free_queue = []
    coach_full_queue = []
    for device in range(flags.num_actor_devices):
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        free_queue.append(_free_queue)
        full_queue.append(_full_queue)
        coach_free_queue.append(ctx.SimpleQueue())
        coach_full_queue.append(ctx.SimpleQueue())

    # Learner model for training
    learner_model = Model(device=flags.training_device)
    coach_model = Coach().to(torch.device('cuda:'+str(flags.training_device)))

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model)
    coach_optim = torch.optim.Adam(coach_model.parameters(), lr=flags.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
        'coach_loss',
        'pred_win',
        'pred_win_std',
        'pred_acc',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
                checkpointpath, map_location="cuda:"+str(flags.training_device)
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            coach_model.load_state_dict(checkpoint_states["coach_model_state_dict"])
            coach_optim.load_state_dict(checkpoint_states["coach_optimizer_state_dict"])
            for device in range(flags.num_actor_devices):
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
                coach_actor_models[device].load_state_dict(coach_model.state_dict())
        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.logger.info(f"Resuming preempted job, current stats:\n{stats}")

    # Starting actor processes
    for device in range(flags.num_actor_devices):
        num_actors = flags.num_actors
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, free_queue[device], full_queue[device], coach_free_queue[device], coach_full_queue[device], \
                    models[device], coach_actor_models[device], buffers[device], coach_buffers[device], flags, coach_thresh))
            actor.start()
            actor_processes.append(actor)
    

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            _stats = learn(position, models, learner_model.get_model(position), batch, 
                optimizers[position], flags, position_lock)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B


    def coach_batch_and_learn(device, local_lock, learn_lock, lock=threading.Lock()):
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(coach_free_queue[device], coach_full_queue[device], coach_buffers[device], flags, local_lock)
            _stats = learn_coach(coach_actor_models, coach_model, batch, coach_optim, criterion, flags, learn_lock)
            with lock:
                for k in _stats:
                    stats[k] = _stats[k]

    for device in range(flags.num_actor_devices):
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)
            coach_free_queue[device].put(m)

    threads = []
    locks = [{'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()} for _ in range(flags.num_actor_devices)]
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}
    coach_lock = threading.Lock()
    coach_learn_lock = threading.Lock()
    for device in range(flags.num_actor_devices):
        coa_thread = ExceptionThread(target=coach_batch_and_learn, args=(device, coach_lock, coach_learn_lock))
        coa_thread.start()
        threads.append(coa_thread)
        for i in range(flags.num_threads):
            for position in ['landlord', 'landlord_up', 'landlord_down']:
                thread = ExceptionThread(
                    target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,device,position,locks[device][position],position_locks[position]))
                thread.start()
                threads.append(thread)
    
    
    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.logger.info('Saving checkpoint to %s', checkpointpath)
        _models = learner_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'coach_model_state_dict':coach_model.state_dict(),
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            'coach_optimizer_state_dict':coach_optim.state_dict(),
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
            coach_model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, 'coach_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)
            torch.save(coach_model.state_dict(), coach_model_weights_dir)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        initial_time = timer() - flags.save_interval * 60
        flag = [0 for _ in range(5)]
        while frames < flags.total_frames:
            start_frames = frames
            steps = frames / 3200
            if steps > 20000 and steps< 40000 and flag[0] == 0:
                with coach_thresh.get_lock():
                    coach_thresh.value = 0.1
                flag[0] = 1
            elif steps > 40000 and steps < 60000 and flag[1] == 0:
                with coach_thresh.get_lock():
                    coach_thresh.value = 0.15
                flag[1] = 1
            elif steps > 60000 and steps < 80000 and flag[2] == 0:
                with coach_thresh.get_lock():
                    coach_thresh.value = 0.2
                flag[2] = 1
            elif steps > 80000 and steps < 100000 and flag[3] == 0:
                with coach_thresh.get_lock():
                    coach_thresh.value = 0.25
                flag[3] = 1
            elif steps > 100000 and flag[4] == 0:
                with coach_thresh.get_lock():
                    coach_thresh.value = 0.3
                flag[4] = 1
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)
            print(coach_thresh.value)
            # if timer() - last_checkpoint_time > flags.save_interval * 60:  
            #     checkpoint(frames)
            #     test_time = timer() - initial_time
            #     last_checkpoint_time = timer()
            #     os.system("python3 generate_eval_data.py --num_games 10000")
            #     time.sleep(10)
            #     os.system("python3 /root/doudizhu/DouZero/ADP_test.py --time " + str(test_time) + " --frames " + str(frames) + " &")
            #     time.sleep(10)
            #     os.system("python3 /root/doudizhu/DouZero/sl_test.py --time " + str(test_time) + " --frames " + str(frames) + " &")

            end_time = timer()
            fps = (frames - start_frames) / (end_time - start_time)
            position_fps = {k:(position_frames[k]-position_start_frames[k])/(end_time-start_time) for k in position_frames}
            log.logger.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                         frames,
                         position_frames['landlord'],
                         position_frames['landlord_up'],
                         position_frames['landlord_down'],
                         fps,
                         position_fps['landlord'],
                         position_fps['landlord_up'],
                         position_fps['landlord_down'],
                         pprint.pformat(stats))

    except KeyboardInterrupt:
        return 
    else:
        for thread in threads:
            thread.join()
        log.logger.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
    