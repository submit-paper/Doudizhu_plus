import os 
import typing
import _pickle as pickle
import logging
from logging import handlers
import sys
import traceback
import numpy as np
from collections import Counter
import time
from copy import deepcopy

import torch 
from torch import multiprocessing as mp

from .env_utils import Environment
from douzero.env import Env

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

def create_optimizers(flags, learner_model, predict_model):
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            [{'params': learner_model.parameters(position)}, {'params': predict_model.parameters(position)}],
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
                hand_legal=dict(size=(T, 15, 5), dtype=torch.float32),
                down_label=dict(size=(T, 15), dtype=torch.int8)
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


def act(i, device, free_queue, full_queue, coach_free_queue, coach_full_queue, pre_model, model, coach_model, buffers, \
    coach_buffers, flags, coach_thresh, share_lock):
    positions = ['landlord', 'landlord_up', 'landlord_down']
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
        hand_legal_buf = {p: [] for p in positions}
        down_label_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        init_landlord_buf = []
        init_landlord_down_buf = []
        init_landlord_up_buf = []
        win_buf = []
        win_pred_buf = []

        size_coach = 0
        total_count = 0

        position, obs, env_output = env.initial()
        
        while True:
            init_landlord = obs['init_landlord'].unsqueeze(0)
            init_landlord_up = obs['init_landlord_up'].unsqueeze(0)
            init_landlord_down = obs['init_landlord_down'].unsqueeze(0)
            share_lock.acquire()
            gate = coach_thresh.value
            share_lock.release()
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
                while True:
                    hand_legal_buf[position].append(obs['hand_legal'])
                    down_label_buf[position].append(obs['down_label'])
                    obs_x_no_action_buf[position].append(env_output['obs_x_no_action'])
                    obs_z_buf[position].append(env_output['obs_z'])
                    with torch.no_grad():
                        if len(env_output['obs_z'].size()) == 2:
                            env_output['obs_z'] = env_output['obs_z'].unsqueeze(0)
                        if len(env_output['obs_x_no_action'].size()) == 1:
                            env_output['obs_x_no_action'] = env_output['obs_x_no_action'].unsqueeze(0)
                        logits, prob = pre_model.forward(position, env_output['obs_z'], env_output['obs_x_no_action'],
                                                         obs['hand_legal'])
                        prob = prob.view(1, -1)
                        predict_hand = prob.expand(obs['x_batch'].shape[0], -1)
                        agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], predict_hand,
                                                     flags=flags)
                        # agent_output = model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags)
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
                        total_count += 1
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
                        buffers[p]['hand_legal'][index][t, ...] = hand_legal_buf[p][t]
                        buffers[p]['down_label'][index][t, ...] = down_label_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    hand_legal_buf[p] = hand_legal_buf[p][T:]
                    down_label_buf[p] = down_label_buf[p][T:]
                    size[p] -= T
            if total_count % 10000 == 1:
                log.logger.info('The threshold of Coach is %.2f ', gate)

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


