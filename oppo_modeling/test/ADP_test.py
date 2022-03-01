import time
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='DouDizhu Evaluation')
    parser.add_argument('--time', default=0.0, type=float)
    parser.add_argument('--frames', default=0, type=int)
    return parser
    

def judge():
    p = os.popen("ps aux | grep generate_eval_data.py | grep -v grep | awk '{print $2}' ")
    x = p.read()
    print(x)
    p.close()
    if not x:        
        return True
    else:
        return False
        
def judge_test():
    p = os.popen("ps aux | grep baseline_ADP | grep -v grep | awk '{print $2}' ")
    x = p.read()
    print(x)
    p.close()
    if not x:        
        return True
    else:
        return False


if __name__ == '__main__':
    args = get_parser().parse_args()
    os.system("sh get_most_recent.sh douzero_checkpoints/douzero/")
    time.sleep(3)
    #os.system("python3 generate_eval_data.py --num_games 10000")
    #time.sleep(3)
    #flag = judge()
    #while not flag:
        #time.sleep(3)
        #flag = judge()
    os.system("python3 evaluate.py --landlord most_recent_model/landlord.ckpt --landlord_up baseline_ADP --landlord_down baseline_ADP --gpu_device 3 --time " + str(args.time) + " --frames " + str(args.frames) + " >> ADP_test.log")
    flag1 = judge_test()
    while not flag1:
        time.sleep(30)
        flag1 = judge_test()
    os.system("python3 evaluate.py --landlord baseline_ADP --landlord_up most_recent_model/landlord_up.ckpt --landlord_down most_recent_model/landlord_down.ckpt --gpu_device 3 --time " + str(args.time) + " --frames " + str(args.frames) + " >> ADP_test.log")
   
    
