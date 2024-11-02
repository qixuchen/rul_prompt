'''
Network definition according to Zhenghua's Keras code
By Ruibing
'''
import torch.nn as nn
import torch
from collections import OrderedDict
from einops import repeat
from .trans_base import Transformer
import numpy as np

class TST(nn.Module):
    def __init__(self, in_length, in_dim, embed_dim, dim, depth, heads, mlp_dim, pool = 'reg', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        assert pool in {'reg', 'mean'}, 'pool type must be either reg (reg token) or mean (mean pooling)'

        self.pos_embedding = nn.Parameter(torch.randn(1, in_length + 1, embed_dim))
        # self.pos_embedding = torch.tensor(self.gen_pe(len_seq = in_length + 1), requires_grad=False).cuda()

        self.embedding = nn.Linear(in_dim, embed_dim)
        self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.proj = nn.Linear(embed_dim, dim)
        self.bn = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )
    
    def gen_pe(self, len_seq, p_dim = 16):

        def get_pe(p):
            return [p / np.power(10000, 2 * (hid_j // 2) / p_dim) for hid_j in range(p_dim)]
        
        pe_table = np.array([[get_pe(p_i) for p_i in range(len_seq)]])
        pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])  # dim 2i
        pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])  # dim 2i+1

        return pe_table

    def forward(self, x, aux):
        # x shape: (N, L, H_in)
        # x_tst = torch.transpose(x, 1, 2)
        x = self.embedding(x)
        # x: n, l, d
        n, l, _ = x.shape

        reg_tokens = repeat(self.reg_token, '() l d -> n l d', n = n)
        x = torch.cat((reg_tokens, x), dim=1)

        x += self.pos_embedding[:, :(l + 1)]
        x = self.dropout(x)

        x = self.proj(x)
        x = self.bn(x)
        x= self.relu(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)