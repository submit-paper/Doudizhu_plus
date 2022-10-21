import torch
import numpy as np

from douzero.env.env import test_get_obs

def _load_model(position, model_path):
    from douzero.dmc.models import model_dict, pre_model_dict
    middle_path = model_path.split('/')
    middle_path[1] = '/pre_' + middle_path[1]
    pre_model_path = "".join(middle_path)
    model = model_dict[position]()
    model_state_dict = model.state_dict()
    pred_model = pre_model_dict[position]()
    pred_model_state_dict = pred_model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
        pred_pretrained = torch.load(pre_model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
        pred_pretrained = torch.load(pre_model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    pred_pretrained = {k: v for k, v in pred_pretrained.items() if k in pred_model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    pred_model_state_dict.update(pred_pretrained)
    pred_model.load_state_dict(pred_model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
        pred_model.cuda()
    model.eval()
    pred_model.eval()
    return model, pred_model

class DeepAgent:

    def __init__(self, position, model_path):
        self.model, self.pre_model = _load_model(position, model_path)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = test_get_obs(infoset)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        obs_z = torch.from_numpy(obs['z']).float()
        obs_x = torch.from_numpy(obs['x_no_action']).float()
        if len(obs_z.size()) == 2:
            obs_z = obs_z.unsqueeze(0)
        if len(obs_x.size()) == 1:
            obs_x = obs_x.unsqueeze(0)
        hand_legal = obs['hand_legal']
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
            obs_z, obs_x = obs_z.cuda(), obs_x.cuda()
            hand_legal = hand_legal.cuda()
        _, pred_hand = self.pre_model.forward(obs_z, obs_x, hand_legal)
        prob = pred_hand.view(1, -1)
        predict_hand = prob.expand(x_batch.shape[0], -1)
        y_pred = self.model.forward(z_batch, x_batch, predict_hand, return_value=True)['values']
        # y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
