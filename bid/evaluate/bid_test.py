import pickle
import os
import time
import numpy as np
import argparse
from douzero.bid.evaluation import evaluate


if __name__ == '__main__':
    #os.system("python3 generate_eval_data.py --num_games 10000")
    time.sleep(5)
    parser = argparse.ArgumentParser(
                    'Dou Dizhu Bid')
    parser.add_argument('--landlord', type=str,
            default='baselines/douzero_WP/landlord.ckpt')
    parser.add_argument('--landlord_up', type=str,
            default='baselines/WP/landlord_up.ckpt')
    parser.add_argument('--landlord_down', type=str,
            default='baselines/WP/landlord_down.ckpt')
    parser.add_argument('--eval_data', type=str,
            default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--gpu_device', type=str, default='0')

    args = parser.parse_args()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    paras = [0, 0, 0]   # The three threshold values that determines the bidding
    evaluate(args.landlord,
             args.landlord_up,
             args.landlord_down,
             args.eval_data,
             args.num_workers,
             paras)

