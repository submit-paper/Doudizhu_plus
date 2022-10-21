import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MystateDataset
import numpy as np
from tqdm import tqdm
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt


class Bid(nn.Module):
    def __init__(self):
        super().__init__()
        self.bid_embed = nn.Embedding(31, 256)
        self.bid_fc1 = nn.Linear(256, 128)
        self.bid_fc2 = nn.Linear(128, 128)
        self.bid_fc3 = nn.Linear(128, 128)
        self.bid_fc4 = nn.Linear(128, 64)
        self.bid_fc5 = nn.Linear(64, 1)

    def forward(self, landlord):
        x = self.bid_embed(landlord)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.bid_fc1(x))
        x = torch.relu(self.bid_fc2(x))
        x = torch.relu(self.bid_fc3(x))
        x = torch.relu(self.bid_fc4(x))
        out = torch.sigmoid(self.bid_fc5(x))
        return out


def testresult(test_loader, network):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss().to(device)
    with torch.no_grad():
        acc_list = []
        for batch_idx, (handpai, ratio) in enumerate(test_loader):
            handpai = handpai.to(device)
            ratio = ratio.unsqueeze(-1).to(device)
            res = network(handpai)
            loss = criterion(res, ratio)
            acc_list.append(loss.item())
        acc = np.mean(acc_list)
    return acc


def train(epochs, trainfile, testfile):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Bid().to(device)
    # net = nn.DataParallel(net)
    criterion = torch.nn.MSELoss().to(device)

    batchsize = 4096

    optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)
    test_data = MystateDataset(datapkl=testfile)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batchsize, shuffle=True, num_workers=0)
    train_data = MystateDataset(datapkl=trainfile)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batchsize, shuffle=True, num_workers=0)

    loss_pic = []
    train_epoch = []
    test_epoch = []
    test_pic = []
    for epoch in tqdm(range(1, epochs + 1)):
        loss_list = []
        for batch_idx, (handpai, ratio) in enumerate(train_loader):
            handpai = handpai.to(device)
            ratio = ratio.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            res = net(handpai)
            loss = criterion(res, ratio)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        mean_loss = np.mean(loss_list)
        train_epoch.append(epoch)
        loss_pic.append(mean_loss)
        print('train', mean_loss)
        if epoch % 2 == 0:
            acc = testresult(test_loader, net)
            test_epoch.append(epoch)
            test_pic.append(acc)
            print('test', acc)
    torch.save(net.state_dict(), './model/bid.pkl')
    plt.subplot(2, 1, 1)
    plt.plot(train_epoch, loss_pic)
    plt.ylabel('train loss')
    plt.subplot(2, 1, 2)
    plt.plot(test_epoch, test_pic, '.-')
    plt.ylabel('test acc')
    plt.show()


if __name__ == '__main__':
    train(120, './train.pkl', './test.pkl')
