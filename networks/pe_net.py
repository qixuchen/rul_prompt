'''
Network definition according to Zhenghua's Keras code
By Ruibing
'''
import torch.nn as nn
import torch
from collections import OrderedDict
import numpy as np

EPSILON = 1e-8

class PE_NET(nn.Module):
    def __init__(self, input_dim):
        super(PE_NET, self).__init__()
        print('Load the pe_net.')

        self.cnn_base = nn.Conv1d(14, 16, 1, padding=0)
        self.pe_encode = nn.Conv1d(16, 16, 1, bias=False)
        
        # self.cnn_batchnorm = nn.BatchNorm1d(16)
        self.cnn_layernorm = nn.LayerNorm([16, 30])
        
        self.cnn_relu = nn.ReLU(inplace=True)

        self.cnn_backbone = nn.Sequential(OrderedDict([
                                ('cnn1', nn.Conv1d(in_channels = 16, out_channels = 16, kernel_size = 5, stride=2, padding=2)),
                                ('ln1', nn.LayerNorm([16, 15])),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('cnn2_1', nn.Conv1d(16, 64, 3, padding=1)),
                                ('ln2_1', nn.LayerNorm([64, 15])),
                                ('relu2_1', nn.ReLU(inplace=True)),
                                ('cnn2_2', nn.Conv1d(64, 64, 3, padding=1)),
                                ('ln2_2', nn.LayerNorm([64, 15])),
                                ('relu2_2', nn.ReLU(inplace=True)),
                                ('cnn3_1', nn.Conv1d(64, 128, 3, 2, padding=1)),
                                ('ln3_1', nn.LayerNorm([128, 8])),
                                ('relu3_1', nn.ReLU(inplace=True)),
                                ('cnn3_2', nn.Conv1d(128, 128, 3, 1, padding=1)),
                                ('ln3_2', nn.LayerNorm([128, 8])),
                                ('relu3_2', nn.ReLU(inplace=True)),
                                ('cnn4_1', nn.Conv1d(128, 256, 3, 2, padding=1)),
                                ('ln4_1', nn.LayerNorm([256, 4])),
                                ('relu4_1', nn.ReLU(inplace=True)),
                                ]))

        self.fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 256)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(256, 256)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=0.2))
                                ]))

        self.reg = nn.Linear(256, 1)

    # Defining the forward pass
    def forward(self, x, aux, val=False):
        # x shape: (N, L, H_in)
        # aux shape: (N, L, H_in)

        x_trans = torch.transpose(x, 1, 2)
        x_trans = torch.split(x_trans, [14, 16], 1)

        x_feat = self.cnn_base(x_trans[0])
        x_pe = self.pe_encode(x_trans[1])
        x_cnn = x_feat + x_pe
        x_cnn = self.cnn_layernorm(x_cnn)
        x_cnn = self.cnn_relu(x_cnn)

        x_cnn = self.cnn_backbone(x_cnn)
        x_cnn = x_cnn.view(x_cnn.shape[0],-1)

        x_cnn = self.fc(x_cnn)
        out = self.reg(x_cnn)
        
        return out