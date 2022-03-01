import os
import threading
import time
import timeit
import pprint
import logging
import sys
import traceback
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model, Coach
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act, create_coach_buffers


mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}


class ExceptionThread(threading.Thread):           # Log the condition of multithreading

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


def compute_acc(pred, label):   # Compute the accuracy of coach nework for log
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

def learn_coach(coach_actor_models, model, batch, optimizer, criterion, flags, lock):  # Update the coach network, the implementation refers to original DouZero
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

def train(flags):        # The implementation of coach network refers to that of DouZero so that much code is similar 
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size
    coach_thresh = mp.Manager().Value('f', 0)
    share_lock = mp.Manager().Lock()
    
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
                    models[device], coach_actor_models[device], buffers[device], coach_buffers[device], flags, coach_thresh, share_lock))
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


    def coach_batch_and_learn(device, local_lock, learn_lock, lock=threading.Lock()):   # Update the coach network
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
        while frames < flags.total_frames:        # The threshold is determinded by the training steps.
            start_frames = frames
            steps = frames / 3200
            if steps > 20000 and steps< 40000 and flag[0] == 0:
                share_lock.acquire()
                coach_thresh.value = 0.1
                share_lock.release()  
                flag[0] = 1
            elif steps > 40000 and steps < 60000 and flag[1] == 0:
                share_lock.acquire()
                coach_thresh.value = 0.15
                share_lock.release()  
                flag[1] = 1
            elif steps > 60000 and steps < 80000 and flag[2] == 0:
                share_lock.acquire()
                coach_thresh.value = 0.2
                share_lock.release()  
                flag[2] = 1
            elif steps > 80000 and steps < 100000 and flag[3] == 0:
                share_lock.acquire()
                coach_thresh.value = 0.25
                share_lock.release()  
                flag[3] = 1
            elif steps > 100000 and flag[4] == 0:
                share_lock.acquire()
                coach_thresh.value = 0.3
                share_lock.release()  
                flag[4] = 1
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)
            #print(coach_thresh.value)
            if timer() - last_checkpoint_time > flags.save_interval * 60:   # When new models are saved, start a round of test
                checkpoint(frames)
                test_time = timer() - initial_time
                last_checkpoint_time = timer()
                os.system("python3 generate_eval_data.py --num_games 10000")
                time.sleep(10)
                os.system("python3 /root/doudizhu/DouZero/ADP_test.py --time " + str(test_time) + " --frames " + str(frames) + " &")
                time.sleep(10)
                os.system("python3 /root/doudizhu/DouZero/sl_test.py --time " + str(test_time) + " --frames " + str(frames) + " &")

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
    # os.system("ps aux | grep mempool.py | grep -v grep | awk '{print $2}' | xargs kill -9")
