'''
Network definition according to Zhenghua's Keras code
By Ruibing
'''
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict

EPSILON = 1e-8

class CNN_B_CLIP(nn.Module):
    def __init__(self, num_hidden, input_dim, aux_dim, prompt_dict):
        super(CNN_B_CLIP, self).__init__()
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.aux_dim = aux_dim
        self.prompt_dict = torch.tensor(prompt_dict).float().cuda()

        self.cnn_basic = nn.Sequential(OrderedDict([
                                ('cnn1', nn.Conv1d(in_channels = 14, out_channels = 16, kernel_size = 5, stride=2, padding=2)),
                                ('bn1', nn.BatchNorm1d(16)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('cnn2_1', nn.Conv1d(16, 64, 3, padding=1)),
                                ('bn2_1', nn.BatchNorm1d(64)),
                                ('relu2_1', nn.ReLU(inplace=True)),
                                ('cnn2_2', nn.Conv1d(64, 64, 3, padding=1)),
                                ('bn2_2', nn.BatchNorm1d(64)),
                                ('relu2_2', nn.ReLU(inplace=True)),
                                ('cnn3_1', nn.Conv1d(64, 128, 3, 2, padding=1)),
                                ('bn3_1', nn.BatchNorm1d(128)),
                                ('relu3_1', nn.ReLU(inplace=True)),
                                ('cnn3_2', nn.Conv1d(128, 128, 3, 1, padding=1)),
                                ('bn3_2', nn.BatchNorm1d(128)),
                                ('relu3_2', nn.ReLU(inplace=True)),
                                ('cnn4_1', nn.Conv1d(128, 256, 3, 2, padding=1)),
                                ('bn4_1', nn.BatchNorm1d(256)),
                                ('relu4_1', nn.ReLU(inplace=True)),
                                ]))

        self.fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 256)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(256, 256)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=0.2)),
                                ('fc3', nn.Linear(256, 128)),
                                ('relu3', nn.ReLU(inplace=True))
                                ]))

        self.pmp_proj1 = nn.Linear(512, 128)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # Defining the forward pass
    def forward(self, x, aux, pmp1, val=False):
        # x shape: (N, L, H_in)
        # aux shape: (N, L, H_in)

        x_cnn = torch.transpose(x, 1, 2)
        x_cnn = self.cnn_basic(x_cnn)
        
        x_cnn = x_cnn.view(x_cnn.shape[0],-1)
        x = self.fc(x_cnn)

        x = x / (x.norm(dim=1, keepdim=True) + EPSILON)
        
        pmp_feat1 = self.pmp_proj1(pmp1)

        pmp_feat1 = pmp_feat1 / (pmp_feat1.norm(dim=1, keepdim=True) + EPSILON)

        logit_scale = self.logit_scale.exp()
        logits_per_x = logit_scale * x @ pmp_feat1.t()
        # logits_per_x = x @ pmp_feat1.t()
        logits_per_pmp1 = logits_per_x.t()

        if val:
            preds_feat = self.pmp_proj1(self.prompt_dict)
            preds_feat = preds_feat / (preds_feat.norm(dim=1, keepdim=True) + EPSILON)
            logits_per_pred = logit_scale * x @ preds_feat.t()
            # logits_per_pred = x @ preds_feat.t()
            
        if not val:
            return logits_per_x, logits_per_pmp1
        else:
            return logits_per_x, logits_per_pmp1, logits_per_pred