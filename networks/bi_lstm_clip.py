'''
Network definition according to Zhenghua's Keras code
By Ruibing
'''
import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np

EPSILON = 1e-8

class Bi_LSTM_CLIP(nn.Module):
    def __init__(self, input_dim, prompt_dict):
        super(Bi_LSTM_CLIP, self).__init__()

        self.input_dim = input_dim
        self.prompt_dict = torch.tensor(prompt_dict).float().cuda()

        self.bi_lstm1 = nn.LSTM(input_size  = self.input_dim,
                                hidden_size = 16,
                                num_layers = 1,
                                batch_first = True,
                                dropout = 0,
                                bidirectional = True)
        self.drop1 = nn.Dropout(p=0.2)

        self.bi_lstm2 = nn.LSTM(input_size  = 16,
                        hidden_size = 32,
                        num_layers = 1,
                        batch_first = True,
                        dropout = 0,
                        bidirectional = True)
        self.drop2 = nn.Dropout(p=0.2)

        self.fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(32, 16)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('fc2', nn.Linear(16, 8)),
                                ('relu2', nn.ReLU(inplace=True))
                                ]))

        # self.cls = nn.Linear(8, 1)

        self.pmp_proj1 = nn.Linear(512, 8)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # Defining the forward pass
    def forward(self, x, aux, pmp1, val=False):
        
        # import ipdb
        # ipdb.set_trace()
        # x: 10,30,14
        x, hidden = self.bi_lstm1(x)
        x_split = torch.split(x, (x.shape[2]//2), 2)
        x = x_split[0] + x_split[1]
        x = self.drop1(x)

        x, hidden = self.bi_lstm2(x)
        x_split = torch.split(x, (x.shape[2]//2), 2)
        x = x_split[0] + x_split[1]
        x = self.drop2(x)

        x = x.split(1,1)[0]
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        x = x / (x.norm(dim=1, keepdim=True) + EPSILON)
        
        pmp_feat1 = self.pmp_proj1(pmp1)

        pmp_feat1 = pmp_feat1 / (pmp_feat1.norm(dim=1, keepdim=True) + EPSILON)

        logit_scale = self.logit_scale.exp()
        # logits_per_x = logit_scale * x @ pmp_feat1.t()
        logits_per_x = x @ pmp_feat1.t()
        logits_per_pmp1 = logits_per_x.t()

        if val:
            preds_feat = self.pmp_proj1(self.prompt_dict)
            preds_feat = preds_feat / (preds_feat.norm(dim=1, keepdim=True) + EPSILON)
            logits_per_pred = logit_scale * x @ preds_feat.t()
            
        if not val:
            return logits_per_x, logits_per_pmp1
        else:
            return logits_per_x, logits_per_pmp1, logits_per_pred