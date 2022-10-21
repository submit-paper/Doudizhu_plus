import torch
import numpy as np

from douzero.env.env import test_get_obs

def _load_model(position, model_type):
    from douzero.dmc.models import baseline_model_dict
    if model_type == 'baseline_sl':
        model_path = 'baselines/sl/' + position + '.ckpt'
    elif model_type == 'baseline_ADP':
        model_path = 'baselines/douzero_ADP/' + position + '.ckpt'
    else:
        model_path = 'baselines/douzero_WP/' + position + '.ckpt'
    model = baseline_model_dict[position]()
    model_state_dict = model.state_dict()
    if torch.cuda.is_available():
        pretrained = torch.load(model_path, map_location='cuda:0')
    else:
        pretrained = torch.load(model_path, map_location='cpu')
    pretrained = {k: v for k, v in pretrained.items() if k in model_state_dict}
    model_state_dict.update(pretrained)
    model.load_state_dict(model_state_dict)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model

class BaseAgent:

    def __init__(self, position, model_path):
        self.model = _load_model(position, model_path)

    def act(self, infoset):
        if len(infoset.legal_actions) == 1:
            return infoset.legal_actions[0]

        obs = test_get_obs(infoset)

        z_batch = torch.from_numpy(obs['z_batch']).float()
        x_batch = torch.from_numpy(obs['x_batch']).float()
        if torch.cuda.is_available():
            z_batch, x_batch = z_batch.cuda(), x_batch.cuda()
        y_pred = self.model.forward(z_batch, x_batch, return_value=True)['values']
        y_pred = y_pred.detach().cpu().numpy()

        best_action_index = np.argmax(y_pred, axis=0)[0]
        best_action = infoset.legal_actions[best_action_index]

        return best_action
