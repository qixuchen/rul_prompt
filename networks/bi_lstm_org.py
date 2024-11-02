'''
Network definition according to Zhenghua's Keras code
By Ruibing
'''
import torch.nn as nn
import torch
from collections import OrderedDict

class Bi_LSTM(nn.Module):
    def __init__(self, num_hidden, input_dim, aux_dim):
        super(Bi_LSTM, self).__init__()
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.aux_dim = aux_dim

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

        self.cls = nn.Linear(8, 1)

    # Defining the forward pass
    def forward(self, x, aux):
        
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
        out = self.cls(x)

        return out