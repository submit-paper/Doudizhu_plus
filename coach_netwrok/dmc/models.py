import numpy as np

import torch
from torch import nn

class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel

class Model:
    def __init__(self, device=0):
        self.models = {}
        self.models['landlord'] = LandlordLstmModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_up'] = FarmerLstmModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_down'] = FarmerLstmModel().to(torch.device('cuda:'+str(device)))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models


class Coach(nn.Module):
    def __init__(self):
        super().__init__()
        self.landlord_embed = nn.Embedding(31, 256)
        self.landlord_fc1 = nn.Linear(256, 128)
        self.landlord_fc2 = nn.Linear(128, 128)
        self.landlord_fc3 = nn.Linear(128, 128)
        self.landlord_up_embed = nn.Embedding(31, 256)
        self.landlord_up_fc1 = nn.Linear(256, 128)
        self.landlord_up_fc2 = nn.Linear(128, 128)
        self.landlord_up_fc3 = nn.Linear(128, 128)
        self.landlord_down_embed = nn.Embedding(31, 256)
        self.landlord_down_fc1 = nn.Linear(256, 128)
        self.landlord_down_fc2 = nn.Linear(128, 128)
        self.landlord_down_fc3 = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64,1)

    def forward(self, landlord, landlord_down, landlord_up):
        x = self.landlord_embed(landlord)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.landlord_fc1(x))
        x = torch.relu(self.landlord_fc2(x))
        x = torch.relu(self.landlord_fc3(x))

        y = self.landlord_down_embed(landlord_down)
        y = torch.mean(y, dim=1)
        y = torch.relu(self.landlord_down_fc1(y))
        y = torch.relu(self.landlord_down_fc2(y))
        y = torch.relu(self.landlord_down_fc3(y))

        z = self.landlord_up_embed(landlord_up)
        z = torch.mean(z, dim=1)
        z = torch.relu(self.landlord_up_fc1(z))
        z = torch.relu(self.landlord_up_fc2(z))
        z = torch.relu(self.landlord_up_fc3(z))

        data = torch.cat((x, y, z), -1)
        res = torch.relu(self.fc1(data))
        res = torch.relu(self.fc2(res))
        res = torch.relu(self.fc3(res))
        res = self.fc4(res)

        return res
