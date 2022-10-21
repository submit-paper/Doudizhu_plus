import pickle
import os
import numpy as np
import argparse
from douzero.bid.evaluation import evaluate

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


def generate(num=1000):
    _deck = deck.copy()
    res = []
    np.random.shuffle(_deck)
    landlord = _deck[:17]
    landlord.sort()
    other = _deck[17:]
    for _ in range(num):
        np.random.shuffle(other)
        card_play_data = {'landlord': landlord + other[:3],
                        'landlord_up': other[3:20],
                        'landlord_down': other[20:37],
                        'three_landlord_cards': other[:3],
                        }
        for key in card_play_data:
            card_play_data[key].sort()
        res.append(card_play_data)
    return landlord, res


def record(args):            # Record the simulated strength for the given initial 17 cards
    landlord, data = generate(args.games)
    output_pickle = args.eval_data
    print("saving pickle file...")
    with open(output_pickle,'wb') as g:
        pickle.dump(data,g,pickle.HIGHEST_PROTOCOL)
    
    win_nums = evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
    win_rate = win_nums / args.games
             
    print(landlord, win_rate)
    return landlord, win_rate


def check(args):               # Used for checking how many simulations are suitable
    _deck = deck.copy()
    res = []
    np.random.shuffle(_deck)
    landlord = _deck[:17]
    landlord.sort()
    other = _deck[17:]
    dic = {tuple(landlord):[]}
    for _ in range(10*args.games):
        np.random.shuffle(other)
        card_play_data = {'landlord': landlord + other[:3],
                        'landlord_up': other[3:20],
                        'landlord_down': other[20:37],
                        'three_landlord_cards': other[:3],
                        }
        for key in card_play_data:
            card_play_data[key].sort()
        res.append(card_play_data)
    output_pickle = args.eval_data
    print("saving pickle file...")
    last = 0
    win_bynow = 0
    for num in [args.games * i for i in range(1, 11)]:
        with open(output_pickle,'wb') as g:
            pickle.dump(res[last:num],g,pickle.HIGHEST_PROTOCOL)
        win_nums = evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers)
        win_bynow += win_nums
        win_rate = win_bynow / num
        last = num
        dic[tuple(landlord)].append(win_rate)
    print(dic)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Bid')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_ADP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/ADP/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/ADP/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--output', type=str, default='record.pkl')
    parser.add_argument('--games', type=int, default=5000)
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    #for i in range(1000):
    #    check(args)
    data = []
    for _ in range(args.samples):
        landlord, win_rate = record(args)
        data.append([landlord, win_rate])
    
    record_pickle = args.output
    with open(record_pickle,'wb') as g:
        pickle.dump(data,g,pickle.HIGHEST_PROTOCOL)




