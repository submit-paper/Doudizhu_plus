import torch
from torch.utils import data
import _pickle as pickle


class MystateDataset(data.Dataset):
    def __init__(self, datapkl):
        super(MystateDataset, self).__init__()
        with open(datapkl, 'rb') as f:
            input = pickle.load(f)
        self.input = input

    def __getitem__(self, index):
        handpai = self.input[index][0]
        handpai = torch.as_tensor(handpai)
        ratio = self.input[index][1]
        ratio = torch.as_tensor(ratio)

        return handpai, ratio

    def __len__(self):
        return len(self.input)

