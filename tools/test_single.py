'''
This is Pytorch version code for blstm based on Zhenghua's Keras code
By Ruibing from 2021.05.11
'''
import os
import argparse
import pprint
from datetime import date
import time
import _init_paths

import numpy as np
import random
import warnings
import torch


from config import config, update_config
import create_logger
from lib import metric

from core.model import model
from dataset import data_loader_v2_single_test as data_loader
# from core.model_mov import model
# from dataset import data_loader_mov2_test as data_loader

# from networks.bi_lstm_ops_old import Bi_LSTM
from networks.bi_lstm_ops import Bi_LSTM
# from networks.bi_lstm_org import Bi_LSTM
from networks.bi_lstm_handcraft import Bi_LSTM_HAND
from networks.cnn_bi_lstm import CNN_Bi_LSTM
from networks.cnn_0 import CNN
from networks.cnn_b6_pe import CNN_PE
from networks.cnn_b6 import CNN_B
from networks.cnn_s1 import CNN_S
from networks.cnn_dcn import CNN_DCN
from networks.trans_1 import TST
from networks.bi_lstm_mov2 import Bi_LSTM_MOV
# from networks.cnn_bi_lstm_mov2 import Bi_LSTM_MOV


# torch.backends.cudnn.enabled = False

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
    vis = True

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

    data_iter = data_loader.CMPDataIter(config.data.root, config.data.set, config.data.max_rul, config.data.seq_len, config.net.name, config.data.test_id)

    alldata_loader = torch.utils.data.DataLoader(data_iter, batch_size=config.train.batch_size, num_workers=config.data.num_worker,
                                                pin_memory=True, worker_init_fn=data_loader.worker_init_fn)
                                                
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
    elif config.net.name == 'tst':
        sym_net = TST(in_length=config.data.seq_len, in_dim=config.net.input_dim, embed_dim=16, dim=128, 
                        depth=6, heads=6, mlp_dim=256, pool = 'reg', dim_head = 64, dropout = 0.2, emb_dropout = 0.)
    elif config.net.name == 'blstm_mov':
        sym_net = Bi_LSTM_MOV(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim, hand_dim = config.net.hand_dim)
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
    net.test_load_checkpoint(load_path = config.test.model_path, model_name = config.test.model_name)

    metrics = metric.MetricList(metric.RMSE(max_rul = config.data.max_rul), metric.RULscore(max_rul = config.data.max_rul),)
    net.data_iter = alldata_loader
    net.dataset = data_iter
    net.metrics = metrics

    # test loop:
    net.dataset.reset()
    net.metrics.reset()
    net.net.eval()
    sum_sample_inst = 0
    sum_sample_elapse = 0.
    sum_update_elapse = 0
    net.callback_kwargs['prefix'] = 'Test'
    batch_start_time = time.time()
    if vis:
        rul = []
        gt = []
    for i_batch, dats in enumerate(net.data_iter):

        net.callback_kwargs['batch'] = i_batch
        update_start_time = time.time()
        # [forward] making next step
        outputs, losses = net.forward(dats)

        # [evaluation] update train metric
        metrics.update([output.data.cpu() for output in outputs], dats[-1].cpu(),
                        [loss.data.cpu() for loss in losses])

        # timing each batch
        sum_sample_elapse += time.time() - batch_start_time
        sum_update_elapse += time.time() - update_start_time
        batch_start_time = time.time()
        sum_sample_inst += dats[0].shape[0]

        if (i_batch % net.step_callback_freq) == 0:
            # retrive eval results and reset metic
            net.callback_kwargs['namevals'] = net.metrics.get_name_value()
            # speed monitor
            net.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
            net.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
            sum_update_elapse = 0
            sum_sample_elapse = 0
            sum_sample_inst = 0
            # callbacks
            net.step_end_callback()
        
        #record RUL results for visulization
        if vis:
            rul.extend(outputs[0].cpu().numpy()[:,0].tolist())
            gt.extend(dats[-1].numpy()[:,0].tolist())
            res = {'rul': rul, 'gt': gt}
    
    # retrive eval results and reset metic
    net.callback_kwargs['namevals'] = net.metrics.get_name_value()
    # speed monitor
    net.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
    net.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
    # callbacks
    net.step_end_callback()
    if vis:
        import pickle
        res_pt = 'res/{:}_test_engine{:}.pkl'.format(config.data.set, config.data.test_id)
        with open(res_pt, 'wb') as f:
            pickle.dump(res, f)
        print("Save the test results at {:}".format(res_pt))