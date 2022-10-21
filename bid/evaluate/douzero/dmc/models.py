import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128 + 75, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, pred, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out, x, pred], dim=-1)
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
        self.dense1 = nn.Linear(484 + 128 + 75, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, pred, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x, pred], dim=-1)
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


class LandlordpredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 10**7
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(319 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.pre1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre3 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre4 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre5 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre6 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre7 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre8 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre9 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre10 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre11 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre12 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre13 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.presmall = nn.Linear(512, 5)
        self.prebig = nn.Linear(512, 5)

    def forward(self, z, x, legal):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))
        pre_layer = [self.pre3, self.pre4, self.pre5, self.pre6, self.pre7, self.pre8, self.pre9, self.pre10,
                     self.pre11, self.pre12, self.pre13, self.pre1, self.pre2, self.presmall, self.prebig]
        if len(x.size()) == len(legal.size()):
            legal = legal.unsqueeze(0)
            legal_hand = legal.expand(x.size(-2), -1, -1)
            res = []
            for i in range(len(pre_layer)):
                layer = pre_layer[i]
                x_new = layer(x)
                kind_legal = legal_hand[:, i, :]
                res_layer = x_new - (1 - kind_legal) * self.L
                res.append(res_layer)
            logits = torch.cat(res, dim=0)
        else:
            res = []
            for i in range(len(pre_layer)):
                layer = pre_layer[i]
                x_new = layer(x)
                kind_legal = legal[:, i, :]
                res_layer = x_new - (1 - kind_legal) * self.L
                res.append(res_layer)
            logits = torch.cat(res, dim=-1)
            logits = logits.view(logits.size(0), 15, 5)
        pred = F.softmax(logits, dim=-1)

        return logits, pred


class FarmerpredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 10 ** 7
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(430 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.pre1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre3 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre4 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre5 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre6 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre7 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre8 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre9 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre10 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre11 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre12 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.pre13 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 5))
        self.presmall = nn.Linear(512, 5)
        self.prebig = nn.Linear(512, 5)

    def forward(self, z, x, legal):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = torch.relu(self.dense3(x))
        x = torch.relu(self.dense4(x))

        pre_layer = [self.pre3, self.pre4, self.pre5, self.pre6, self.pre7, self.pre8, self.pre9, self.pre10,
                     self.pre11, self.pre12, self.pre13, self.pre1, self.pre2, self.presmall, self.prebig]
        if len(x.size()) == len(legal.size()):
            legal = legal.unsqueeze(0)
            legal_hand = legal.expand(x.size(-2), -1, -1)
            res = []
            for i in range(len(pre_layer)):
                layer = pre_layer[i]
                x_new = layer(x)
                kind_legal = legal_hand[:, i, :]
                res_layer = x_new - (1 - kind_legal) * self.L
                res.append(res_layer)
            logits = torch.cat(res, dim=0)
        else:
            res = []
            for i in range(len(pre_layer)):
                layer = pre_layer[i]
                x_new = layer(x)
                kind_legal = legal[:, i, :]
                res_layer = x_new - (1 - kind_legal) * self.L
                res.append(res_layer)
            logits = torch.cat(res, dim=-1)
            logits = logits.view(logits.size(0), 15, 5)
        pred = F.softmax(logits, dim=-1)

        return logits, pred


class Base_LandlordLstmModel(nn.Module):
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


class Base_FarmerLstmModel(nn.Module):
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


baseline_model_dict = {}
baseline_model_dict['landlord'] = Base_LandlordLstmModel
baseline_model_dict['landlord_up'] = Base_FarmerLstmModel
baseline_model_dict['landlord_down'] = Base_FarmerLstmModel


model_dict = {}
model_dict['landlord'] = LandlordLstmModel
model_dict['landlord_up'] = FarmerLstmModel
model_dict['landlord_down'] = FarmerLstmModel

pre_model_dict = {}
pre_model_dict['landlord'] = LandlordpredictModel
pre_model_dict['landlord_up'] = FarmerpredictModel
pre_model_dict['landlord_down'] = FarmerpredictModel


class Model:
    def __init__(self, device=0):
        self.models = {}
        self.models['landlord'] = LandlordLstmModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_up'] = FarmerLstmModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_down'] = FarmerLstmModel().to(torch.device('cuda:'+str(device)))

    def forward(self, position, z, x, pred, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, pred, training, flags)

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


class Pre_model:
    def __init__(self, device=0):
        self.models = {}
        self.models['landlord'] = LandlordpredictModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_up'] = FarmerpredictModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_down'] = FarmerpredictModel().to(torch.device('cuda:'+str(device)))

    def forward(self, position, z, x, legal):
        model = self.models[position]
        return model.forward(z, x, legal)

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


class Base_Model:
    def __init__(self, device=0):
        self.models = {}
        self.models['landlord'] = Base_LandlordLstmModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_up'] = Base_FarmerLstmModel().to(torch.device('cuda:'+str(device)))
        self.models['landlord_down'] = Base_FarmerLstmModel().to(torch.device('cuda:'+str(device)))

    def forward(self, position, z, x, return_value=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, return_value, flags)

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
        self.fc4 = nn.Linear(64, 1)

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
