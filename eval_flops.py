'''
This is Pytorch version code for blstm based on Zhenghua's Keras code
By Ruibing from 2021.05.11
'''
import os
import argparse
import pprint
from datetime import date
import _init_paths

import numpy as np
import random
import warnings
import torch
from pthflops import count_ops
from thop import profile
from ptflops import get_model_complexity_info
import torchutils as tu


from config import config, update_config
import create_logger
from lib import metric

from core.model import model
# from core.model_mov import model
# from core.model_mov_late import model

from dataset import data_loader_v2 as data_loader
# from dataset import data_loader_mov2 as data_loader
# from dataset import data_loader_PHM_mov2 as data_loader

from dataset import data_loader_handcraft_v2_1 as data_loader_handcraft

from networks.lstm_deep import LSTM
# from networks.bi_lstm_ops_old import Bi_LSTM
# from networks.bi_lstm_ops import Bi_LSTM
# from networks.bi_lstm_org import Bi_LSTM
# from networks.bi_lstm_huang import Bi_LSTM
# from networks.bi_lstm_pe_huang import Bi_LSTM
# from networks.bi_lstm_fold1 import Bi_LSTM
# from networks.bi_lstm_ghost import Bi_LSTM
from networks.bi_lstm_fold1_5_ghost_c_w2 import Bi_LSTM
# from networks.bi_lstm_fold1_ghost import Bi_LSTM
# from networks.bi_lstm_fold1_ghost_c import Bi_LSTM
from networks.bi_lstm_handcraft import Bi_LSTM_HAND
from networks.cnn_bi_lstm import CNN_Bi_LSTM
from networks.cnn_0 import CNN
from networks.cnn_b6_pe import CNN_PE
# from networks.cnn_b6 import CNN_B
from networks.cnn_b6_1 import CNN_B
from networks.cnn_s1 import CNN_S
from networks.cnn_dcn_2 import CNN_DCN
from networks.trans_1 import TST
from networks.bi_lstm_mov2 import Bi_LSTM_MOV
# from networks.bi_lstm_mov_concat import Bi_LSTM_MOV
from networks.cnn_2d import CNN_2D
# from networks.meta_cnn import META
# from networks.dfcn_1 import META
from networks.dfcn2d_3 import META
# from networks.dfcn2d_res import META
# from networks.meta_cnn_lpe import META_PE
from networks.meta_dcn_pe import META_PE


torch.backends.cudnn.enabled = False

def parse_args():
    parser = argparse.ArgumentParser(description='Code for time series RUL by Ruibing')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    return args

if __name__ == "__main__":

    # set args
    args = parse_args()
    curr_path = os.path.abspath(os.path.dirname(__file__))
    logger, final_output_path, model_fixtime = create_logger.create_logger(curr_path, args.cfg, config)
    config.update({'output_pt': final_output_path})
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    if config.seed:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    if config.net.name == 'cnn_bilstm':
        sym_net = CNN_Bi_LSTM(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name == 'cnn':
        sym_net = CNN(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name == 'cnn_pe':
        sym_net = CNN_PE(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name == 'cnn_b':
        sym_net = CNN_B(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name == 'cnn_s':
        sym_net = CNN_S(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name == 'cnn_dcn':
        sym_net = CNN_DCN(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name == 'cnn_2d':
        sym_net = CNN_2D(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name == 'tst':
        sym_net = TST(in_length=config.data.seq_len, in_dim=config.net.input_dim, embed_dim=16, dim=128, 
                        depth=6, heads=6, mlp_dim=256, pool = 'reg', dim_head = 64, dropout = 0.2, emb_dropout = 0.)
    elif config.net.name == 'blstm_mov':
        sym_net = Bi_LSTM_MOV(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim, hand_dim = config.net.hand_dim)
    elif config.net.name == 'lstm':
        sym_net = LSTM(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name.lower() == 'meta':
        sym_net = META(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    elif config.net.name.lower() == 'meta_pe':
        sym_net = META_PE(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)
    else:
        if config.net.hand_craft:
            sym_net = Bi_LSTM_HAND(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim, hand_dim = config.net.hand_dim)
        else:
            sym_net = Bi_LSTM(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim)

    model_prefix = os.path.join(config.output_pt, 'exp_'+ model_fixtime + '_' + config.net.name)
    net = model(net=sym_net, criterion=torch.nn.MSELoss().cuda(), model_prefix=model_prefix, step_callback_freq=config.train.callback_freq,
            save_checkpoint_freq=config.save_frequency, logger = logger)
    net.net.cuda()
    net.net = torch.nn.DataParallel(net.net).cuda()

    inputs = torch.rand(1,30,14).cuda()
    # count_ops(net.net, inputs, ignore_layers=['split_2'])

    # macs, params = profile(net.net, inputs=(inputs, ))
    # print ('macs: {:}, params: {:}'.format(macs, params))

    # macs, params = get_model_complexity_info(net.net, (30,14), as_strings=True,
    #                                     print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    flops = tu.get_model_flops(net.net, inputs)
    print('FLOPs: {:}'.format(flops))