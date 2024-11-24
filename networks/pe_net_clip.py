'''
Network definition according to Zhenghua's Keras code
By Ruibing
'''
import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np

EPSILON = 1e-8

class PE_NET_CLIP(nn.Module):
    def __init__(self, input_dim, prompt_dict):
        super(PE_NET_CLIP, self).__init__()
        self.prompt_dict = torch.tensor(prompt_dict).float().cuda()
        print('Load the pe_net clip network.')

        self.cnn_base = nn.Conv1d(14, 16, 1, padding=0)
        self.pe_encode = nn.Conv1d(16, 16, 1, bias=False)
        self.cnn_batchnorm = nn.BatchNorm1d(16)
        self.cnn_relu = nn.ReLU(inplace=True)

        self.cnn_backbone = nn.Sequential(OrderedDict([
                                ('cnn1', nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = 5, stride=2, padding=2)),
                                ('bn1', nn.BatchNorm1d(16)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('cnn2_1', nn.Conv1d(16, 64, 3, padding=1)),
                                ('bn2_1', nn.BatchNorm1d(64)),
                                ('relu2_1', nn.ReLU(inplace=True)),
                                # ('cnn2_2', nn.Conv1d(64, 64, 3, padding=1)),
                                # ('bn2_2', nn.BatchNorm1d(64)),
                                # ('relu2_2', nn.ReLU(inplace=True)),
                                # ('cnn3_1', nn.Conv1d(64, 128, 3, 2, padding=1)),
                                # ('bn3_1', nn.BatchNorm1d(128)),
                                # ('relu3_1', nn.ReLU(inplace=True)),
                                # ('cnn3_2', nn.Conv1d(128, 128, 3, 1, padding=1)),
                                # ('bn3_2', nn.BatchNorm1d(128)),
                                # ('relu3_2', nn.ReLU(inplace=True)),
                                # ('cnn4_1', nn.Conv1d(128, 256, 3, 2, padding=1)),
                                # ('bn4_1', nn.BatchNorm1d(256)),
                                # ('relu4_1', nn.ReLU(inplace=True)),
                                ]))

        # self.fc = nn.Sequential(OrderedDict([
        #                         ('fc1', nn.Linear(1024, 256)),
        #                         ('relu1', nn.ReLU(inplace=True)),
        #                         ('dropout1', nn.Dropout(p=0.2)),
        #                         ('fc2', nn.Linear(256, 256)),
        #                         ('relu2', nn.ReLU(inplace=True)),
        #                         ('dropout2', nn.Dropout(p=0.2))
        #                         ]))
        
        self.fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(960, 256)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(256, 8)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=0.2)),
                                ]))
        
        self.pmp_proj1 = nn.Linear(512, 8)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.reg = nn.Linear(256, 1)

    # Defining the forward pass
    def forward(self, x, aux, pmp1, val=False):
        # x shape: (N, L, H_in)
        # aux shape: (N, L, H_in)

        x_trans = torch.transpose(x, 1, 2)
        x_trans = torch.split(x_trans, [14, 16], 1)

        x_feat = self.cnn_base(x_trans[0])
        x_pe = self.pe_encode(x_trans[1])
        x_cnn = x_feat + x_pe
        x_cnn = self.cnn_batchnorm(x_cnn)
        x_cnn = self.cnn_relu(x_cnn)
        x_cnn = self.cnn_backbone(x_cnn)
        x_cnn = x_cnn.view(x_cnn.shape[0],-1)
        
        x_cnn = self.fc(x_cnn)
        x_cnn = x_cnn / (x_cnn.norm(dim=1, keepdim=True) + EPSILON)

        pmp_feat1 = self.pmp_proj1(pmp1)
        pmp_feat1 = pmp_feat1 / (pmp_feat1.norm(dim=1, keepdim=True) + EPSILON)

        logit_scale = self.logit_scale.exp()
        # logits_per_x = logit_scale * x_cnn @ pmp_feat1.t()
        logits_per_x = x_cnn @ pmp_feat1.t()
        logits_per_pmp1 = logits_per_x.t()

        if val:
            preds_feat = self.pmp_proj1(self.prompt_dict)
            preds_feat = preds_feat / (preds_feat.norm(dim=1, keepdim=True) + EPSILON)
            logits_per_pred = logit_scale * x_cnn @ preds_feat.t()

        if not val:
            return logits_per_x, logits_per_pmp1
        else:
            return logits_per_x, logits_per_pmp1, logits_per_pred