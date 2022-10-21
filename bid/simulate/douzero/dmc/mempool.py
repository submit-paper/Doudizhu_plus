import _pickle as pickle
import os, random, time
from copy import deepcopy


def coach_data():
    data_path = "/root/doudizhu/DouZero/data/"
    mempool_data = []
    mempool_size = 3000

    while True:
        count = 0
        filenames = os.listdir(data_path)
        for file in filenames:
            filename = data_path + file
            try:
                f = open(filename, 'rb')
                input_data = pickle.load(f)
                f.close()
                mempool_data += input_data
                count += len(input_data)
                os.remove(filename)
                if len(mempool_data) > mempool_size:
                    mempool_data = mempool_data[-mempool_size : ]
            except:
                pass
        print("mempool_data:", len(mempool_data), "get sample per sec:", count)
        if len(mempool_data) > 64:
            train_data = random.sample(mempool_data, 64)
            with open("/root/doudizhu/DouZero/mempool.pkl", "wb") as fo:
                pickle.dump(train_data, fo)
        time.sleep(1)
