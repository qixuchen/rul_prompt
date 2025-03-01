'''
Network definition according to Zhenghua's Keras code
By Ruibing
'''
import torch.nn as nn
import torch
from collections import OrderedDict

class CNN_B(nn.Module):
    def __init__(self, num_hidden, input_dim, aux_dim):
        super(CNN_B, self).__init__()
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.aux_dim = aux_dim
        print('Load the cnn_b6 defined network.')

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
                                # ('cnn4_2', nn.Conv1d(256, 256, 3, 1, padding=1)),
                                # ('bn4_2', nn.BatchNorm1d(256)),
                                # ('relu4_2', nn.ReLU(inplace=True)),
                                ]))

        self.fc = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(1024, 256)),
                                ('relu1', nn.ReLU(inplace=True)),
                                ('dropout1', nn.Dropout(p=0.2)),
                                ('fc2', nn.Linear(256, 256)),
                                ('relu2', nn.ReLU(inplace=True)),
                                ('dropout2', nn.Dropout(p=0.2))
                                ]))
        
        self.fc_aux = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(120, 256)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('fc2', nn.Linear(256, 256)),
                        ('relu2', nn.ReLU(inplace=True)),
                        ]))
        
        self.fc2 = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(256, 256)),
                        ('relu1', nn.ReLU(inplace=True)),
                        ('dropout1', nn.Dropout(p=0.2)),
                        ]))

        self.reg = nn.Linear(256, 1)

    # Defining the forward pass
    def forward(self, x, aux):
        # x shape: (N, L, H_in)
        # aux shape: (N, L, H_in)

        x_cnn = torch.transpose(x, 1, 2)
        x_cnn = self.cnn_basic(x_cnn)

        # fuse two inputs
        # import ipdb
        # ipdb.set_trace()
        # aux_cnn = aux.view(aux.shape[0],-1)
        # aux_cnn = self.fc_aux(aux_cnn)
        x_cnn = x_cnn.view(x_cnn.shape[0],-1)
        x_cnn = self.fc(x_cnn)

        # x_fuse = x_cnn + aux_cnn
        # x_fuse = self.fc2(x_fuse)

        out = self.reg(x_cnn)

        return out