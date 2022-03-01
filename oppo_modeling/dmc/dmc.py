import os
import threading
import time
import timeit
import pprint
import random
import logging
import sys
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from .file_writer import FileWriter
from .models import Model, Pre_model
from .utils import get_batch, log, create_env, create_buffers, create_optimizers, act

mean_episode_return_buf = {p:deque(maxlen=100) for p in ['landlord', 'landlord_up', 'landlord_down']}

def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets)**2).mean()
    return loss
    
def predict_loss(res, label, criterion):     # Compute the loss of prediction model
    pre = res.reshape(res.size(0)*res.size(1), -1)
    truth = label.reshape(label.size(0)*label.size(1)).long()
    loss = criterion(pre, truth)
    return loss

class ExceptionThread(threading.Thread):   # Log for multithreading

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

def learn(position,    # The prediction models are updated with decision models.
          predict_models,
          pre_model,
          actor_models,
          model,
          batch,
          optimizer,
          flags,
          lock,
          criterion):
    """Performs a learning (optimization) step."""
    device = torch.device('cuda:'+str(flags.training_device)) 
    obs_x_no_action = batch['obs_x_no_action'].to(device)
    obs_action = batch['obs_action'].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))
    hand_legal = torch.flatten(batch['hand_legal'].to(device), 0, 1)
    down_label = torch.flatten(batch['down_label'].to(device), 0, 1)
    obs_x_no_action = torch.flatten(obs_x_no_action, 0, 1)
        
    with lock:
        hand_pred, prob = pre_model(obs_z, obs_x_no_action, hand_legal)
        pre_loss = predict_loss(hand_pred, down_label, criterion)
        handcard = prob.view(prob.size(0), -1)
        downcard = handcard.detach()
        learner_outputs = model(obs_z, obs_x, downcard, return_value=True)
        loss = compute_loss(learner_outputs['values'], target)
        loss += pre_loss
        stats = {
            'mean_episode_return_'+position: torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+position: loss.item(),
            'pre_loss_'+position: pre_loss.item()
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
        nn.utils.clip_grad_norm_(pre_model.parameters(), flags.max_grad_norm)
        optimizer.step()
      
        for actor_model in actor_models:   # Synchronize the learner model with actor models
            actor_model.get_model(position).load_state_dict(model.state_dict())
          
        
        for predict_model in predict_models:
            predict_model.get_model(position).load_state_dict(pre_model.state_dict())
        
        return stats

def train(flags):  
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    # Initialize actor models
    models = []
    pre_models = []
    assert flags.num_actor_devices <= len(flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'
    for device in range(flags.num_actor_devices):
        model = Model(device=device)
        pre_model = Pre_model(device=device)
        model.share_memory()
        pre_model.share_memory()
        model.eval()
        pre_model.eval()
        models.append(model)
        pre_models.append(pre_model)

    # Initialize buffers
    buffers = create_buffers(flags)
   
    # Initialize queues
    actor_processes = []
    ctx = mp.get_context('spawn')
    free_queue = []
    full_queue = []
    for device in range(flags.num_actor_devices):
        _free_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        _full_queue = {'landlord': ctx.SimpleQueue(), 'landlord_up': ctx.SimpleQueue(), 'landlord_down': ctx.SimpleQueue()}
        free_queue.append(_free_queue)
        full_queue.append(_full_queue)

    # Learner model for training
    learner_model = Model(device=flags.training_device)
    predict_model = Pre_model(device=flags.training_device)

    # Create optimizers
    optimizers = create_optimizers(flags, learner_model, predict_model)

    # Stat Keys
    stat_keys = [
        'mean_episode_return_landlord',
        'loss_landlord',
        'pre_loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'pre_loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
        'pre_loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'landlord':0, 'landlord_up':0, 'landlord_down':0}
    
    for k in ['landlord', 'landlord_up', 'landlord_down']:
        for device in range(flags.num_actor_devices):
            models[device].get_model(k).load_state_dict(torch.load('/root/doudizhu/DouZero/most_recent_model/' + k + '0.ckpt'))
            pre_models[device].get_model(k).load_state_dict(torch.load('/root/doudizhu/DouZero/most_recent_model/pre_' + k + '0.ckpt'))

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
                checkpointpath, map_location="cuda:"+str(flags.training_device)
        )
        for k in ['landlord', 'landlord_up', 'landlord_down']:
            learner_model.get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
            predict_model.get_model(k).load_state_dict(checkpoint_states["pre_model_state_dict"][k])
            optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            for device in range(flags.num_actor_devices):
                models[device].get_model(k).load_state_dict(learner_model.get_model(k).state_dict())
                pre_models[device].get_model(k).load_state_dict(predict_model.get_model(k).state_dict())
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
                args=(i, device, free_queue[device], full_queue[device], pre_models[device], models[device], buffers[device], flags))
            actor.start()
            actor_processes.append(actor)

    def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(free_queue[device][position], full_queue[device][position], buffers[device][position], flags, local_lock)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            _stats = learn(position, pre_models, predict_model.get_model(position), models, learner_model.get_model(position), batch, 
                optimizers[position], flags, position_lock, criterion)

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for device in range(flags.num_actor_devices):
        for m in range(flags.num_buffers):
            free_queue[device]['landlord'].put(m)
            free_queue[device]['landlord_up'].put(m)
            free_queue[device]['landlord_down'].put(m)

    threads = []
    locks = [{'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()} for _ in range(flags.num_actor_devices)]
    position_locks = {'landlord': threading.Lock(), 'landlord_up': threading.Lock(), 'landlord_down': threading.Lock()}

    for device in range(flags.num_actor_devices):
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
        pre_models = predict_model.get_models()
        torch.save({
            'model_state_dict': {k: _models[k].state_dict() for k in _models},
            'optimizer_state_dict': {k: optimizers[k].state_dict() for k in optimizers},
            'pre_model_state_dict': {k: pre_models[k].state_dict() for k in pre_models},
            "stats": stats,
            'flags': vars(flags),
            'frames': frames,
            'position_frames': position_frames
        }, checkpointpath)

        # Save the weights for evaluation purpose
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, position+'_weights_'+str(frames)+'.ckpt')))
            pre_model_weights_dir = os.path.expandvars(os.path.expanduser(
                '%s/%s/%s' % (flags.savedir, flags.xpid, 'pre_'+position+'_weights_'+str(frames)+'.ckpt')))
            torch.save(learner_model.get_model(position).state_dict(), model_weights_dir)
            torch.save(predict_model.get_model(position).state_dict(), pre_model_weights_dir)

            
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        initial_time = timer() - flags.save_interval * 60
        last_oppo_time = timer() - flags.oppo_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(10)
            #if timer() - last_oppo_time > flags.oppo_interval * 60:
                #oppopoint()
                #last_oppo_time = timer()

            if timer() - last_checkpoint_time > flags.save_interval * 60:  # When saving new models, start a round of test
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
        log.info('Learning finished after %d frames.', frames)

    checkpoint(frames)
    plogger.close()
