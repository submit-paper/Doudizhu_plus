import pickle as pkl
import random

with open('./record_total.pkl', 'rb') as f:            # Divide the train dataset and test dataset
    a = pkl.load(f)
train = []
test = []
for i in range(len(a)):
    if random.random() <= 0.8:
        train.append(a[i])
    else:
        test.append(a[i])
with open('./train.pkl', 'wb') as f:
    pkl.dump(train, f)
with open('./test.pkl', 'wb') as f:
    pkl.dump(test, f)
