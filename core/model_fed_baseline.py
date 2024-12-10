"""
Author: Qixu Chen
Adapted from data_loader_prompt
"""
import os
from socket import IP_DROP_MEMBERSHIP
import sys
import time
import logging
import json
import random
from copy import deepcopy
from config import config
from collections import OrderedDict

import torch
import numpy as np

sys.path.insert(0, '../lib')
import callback
from lib.utils import select_optimizer

class model(object):

    def __init__(self, net, criterion, config, model_prefix='',
                step_callback=None, epoch_callback=None, logger=None):

        # init params
        self.global_net = net
        self.model_prefix = model_prefix
        self.criterion = criterion
        self.logger = logger
        self.max_rul = 125
        self.test_freq = config.test.test_freq
        self.step_callback_freq = config.train.callback_freq
        self.save_checkpoint_freq = config.save_frequency
        self.n_user = config.fed.n_user
        self.n_user_per_iter = config.fed.n_user_per_iter
        self.train_epoch_per_step = config.fed.train_epoch_per_step
        self.agg_mode = 'all'
        
        self.user_nets = []
        for _ in range(self.n_user):
            self.user_nets.append(deepcopy(self.global_net))
        
        self.callback_kwargs = {'epoch': None, 'batch': None, 'sample_elapse': None, 'update_elapse': None,
                                'epoch_elapse': None, 'namevals': None, 'optimizer_dict': None, 'epoch_num': None,
                                'prefix': None}
        self.epoch_callback_kwargs = {'epoch': None, 'batch': None, 'sample_elapse': None, 'update_elapse': None,
                                'epoch_elapse': None, 'namevals': None, 'optimizer_dict': None,  'epoch_num': None,
                                      'prefix': 'Final'}
        
        if not step_callback:
            step_callback = callback.CallbackList(callback.SpeedMonitor(), callback.MetricPrinter())
        if not epoch_callback:
            # epoch_callback = (lambda **kwargs: None)
            epoch_callback = callback.CallbackList(callback.MetricPrinter())
        
        self.step_callback = step_callback
        self.epoch_callback = epoch_callback


    def step_end_callback(self):
        self.step_callback(**(self.callback_kwargs))


    def epoch_end_callback(self):
        self.epoch_callback(**(self.callback_kwargs))
        if self.callback_kwargs['epoch_elapse'] is not None:
            logging.info("Final_Epoch [{:d}]   time cost: {:.2f} sec ({:.2f} h)".format(
                self.callback_kwargs['epoch'], self.callback_kwargs['epoch_elapse'],
                self.callback_kwargs['epoch_elapse'] / 3600.))
        
        self.epoch_callback(**(self.epoch_callback_kwargs))

        if self.callback_kwargs['epoch'] == 0 or ((self.callback_kwargs['epoch'] + 1) % self.save_checkpoint_freq) == 0:
            self.save_checkpoint(epoch=self.callback_kwargs['epoch'] + 1)
            # self.save_checkpoint(epoch=self.callback_kwargs['epoch'] + 1,
            #                      optimizer_state=self.callback_kwargs['optimizer_dict'])


    # def load_state(self, state_dict, strict=False):
    #     if strict:
    #         self.global_net.load_state_dict(state_dict=state_dict)
    #     else:
    #         # customized partialy load function
    #         net_state_keys = list(self.global_net.state_dict().keys())
    #         for name, param in state_dict.items():
    #             if name in self.global_net.state_dict().keys():
    #                 dst_param_shape = self.global_net.state_dict()[name].shape
    #                 if param.shape == dst_param_shape:
    #                     self.global_net.state_dict()[name].copy_(param.view(dst_param_shape))
    #                     net_state_keys.remove(name)
    #         # indicating missed keys
    #         if net_state_keys:
    #             num_batches_list = []
    #             for i in range(len(net_state_keys)):
    #                 if 'num_batches_tracked' in net_state_keys[i]:
    #                     num_batches_list.append(net_state_keys[i])
    #             pruned_additional_states = [x for x in net_state_keys if x not in num_batches_list]
    #             logging.info("There are layers in current network not initialized by pretrained")
    #             logging.warning(">> Failed to load: {}".format(pruned_additional_states))
    #             return False
    #     return True


    def get_checkpoint_path(self, epoch):
        assert self.model_prefix, "model_prefix undefined!"

        checkpoint_path = "{}_ep-{:04d}.pth".format(self.model_prefix, epoch)

        return checkpoint_path
    
    
    def save_checkpoint(self, epoch, optimizer_state=None):

        save_path = self.get_checkpoint_path(epoch)
        save_folder = os.path.dirname(save_path)

        if not os.path.exists(save_folder):
            logging.debug("mkdir {}".format(save_folder))
            os.makedirs(save_folder)

        if not optimizer_state:
            torch.save({'epoch': epoch, 'state_dict': self.global_net.state_dict()}, save_path)
            logging.info("Checkpoint (only model) saved to: {}".format(save_path))
        else:
            torch.save({'epoch': epoch, 'state_dict': self.global_net.state_dict(), 'optimizer': optimizer_state}, save_path)
            logging.info("Checkpoint (model & optimizer) saved to: {}".format(save_path))


    # def load_checkpoint(self, epoch, optimizer=None):

    #     load_path = self.get_checkpoint_path(epoch)
    #     assert os.path.exists(load_path), "Failed to load: {} (file not exist)".format(load_path)

    #     checkpoint = torch.load(load_path)

    #     all_params_matched = self.load_state(checkpoint['state_dict'], strict=False)

    #     if optimizer:
    #         if 'optimizer' in checkpoint.keys() and all_params_matched:
    #             optimizer.load_state_dict(checkpoint['optimizer'])
    #             logging.info("Model & Optimizer states are resumed from: `{}'".format(load_path))
    #         else:
    #             logging.warning(">> Failed to load optimizer state from: `{}'".format(load_path))
    #     else:
    #         logging.info("Only model state resumed from: `{}'".format(load_path))

    #     if 'epoch' in checkpoint.keys():
    #         if checkpoint['epoch'] != epoch:
    #             logging.warning(">> Epoch information inconsistant: {} vs {}".format(checkpoint['epoch'], epoch))

    # def test_load_checkpoint(self, load_path, model_name, optimizer=None):
        
    #     checkpoint_path = os.path.join(load_path, model_name)
    #     assert os.path.exists(checkpoint_path), "Failed to load: {} (file not exist)".format(checkpoint_path)

    #     checkpoint = torch.load(checkpoint_path)
    #     all_params_matched = self.load_state(checkpoint['state_dict'], strict=True)
    #     logging.info("Load model from: `{}'".format(checkpoint_path))


    # def save_model_json(self, path):
    #     actual_dict = OrderedDict()
    #     for k, v in self.global_net.state_dict().items():
    #         actual_dict[k] = v.tolist()
        
    #     print('Saving model to ', path)
    #     with open(path, 'w') as f:
    #         json.dump(actual_dict, f)
    
    # def load_model_json(self, path):
    #     data_dict = OrderedDict()
    #     with open(path, 'r') as f:
    #         data_dict = json.load(f)    
    #     own_state = self.global_net.state_dict()
    #     for k, v in data_dict.items():
    #         if not k in own_state:
    #             print('Parameter', k, 'not found in own_state!!!')
    #         if type(v) == list or type(v) == int:
    #             v = torch.tensor(v)
    #         own_state[k].copy_(v)
    #     self.global_net.load_state_dict(own_state)
    #     print('Model loaded from ', path)
    
    
    def load_global_state(self, uid):
        target_net = self.user_nets[uid]
        target_net.load_state_dict(state_dict=self.global_net.state_dict())
    
    
    def update_global_net(self, chosen_users):
        
        def average_weights(updated_models):
            weights_avg = deepcopy(updated_models[0])
            for key in weights_avg.keys():
                for i in range(1, len(updated_models)):
                    weights_avg[key] += updated_models[i][key]
                weights_avg[key] = torch.div(weights_avg[key], len(updated_models))
            return weights_avg
        
        updated_models = []
        for uid in chosen_users:
            updated_models.append(self.user_nets[uid].state_dict())
        
        # Update server model based on clients models
        updated_weights = average_weights(updated_models)
        self.global_net.load_state_dict(updated_weights)
        
        
    # def fit(self, data_iter, dataset, optimizer, lr_scheduler, metrics=None, \
    #         epoch_start=0, epoch_end=10000):
    def fit(self, data_iter, dataset, config, metrics=None):
        """
        checking
        """
        assert torch.cuda.is_available(), "only support GPU version"

        """
        start the main loop
        """

        self.data_iter = data_iter
        self.dataset = dataset
        self.metrics = metrics
        self.epoch_start = 0
        self.epoch_end = config.train.end_epoch
        self.callback_kwargs['epoch_num'] = self.epoch_end
        
        
        self.optimizers = []
        self.lr_schedulers = []
        for i in range(len(self.user_nets)):
            self.optimizers.append(select_optimizer(config.train.optimizer, self.user_nets[i].parameters(), config.train.lr))
            self.lr_schedulers.append(torch.optim.lr_scheduler.MultiStepLR(self.optimizers[i], milestones = \
					[int(x) for x in config.train.lr_epoch], gamma=config.train.lr_factor))
        
        self.train_minrmse = None
        self.train_minscore = None
        self.test_minrmse = None
        self.test_minscore = None
        # self.val_minrmse = None
        # self.val_minscore = None
        self.model_id = None


        for i_epoch in range(self.epoch_start, self.epoch_end):
            self.callback_kwargs['epoch'] = i_epoch
            self.epoch_callback_kwargs['namevals'] = []
            epoch_start_time = time.time()

            ###########
            # 1] TRAINING
            ###########
            
            self.train(self.n_user_per_iter)
            
            ###########
            # 2] Evaluation
            ###########
            
            if (self.data_iter is not None) \
                    and (i_epoch % self.test_freq) == 0:
                self.test()
                # self.dataset.reset('val')
                # self.val()

            ###########
            # 2] END OF EPOCH
            ###########
            self.epoch_callback_kwargs['namevals'] += [[('Train_RMSE_cor', self.train_minrmse)]]
            self.epoch_callback_kwargs['namevals'] += [[('Train_RULscore_min', self.train_minscore)]]
            self.epoch_callback_kwargs['namevals'] += [[('Test_RMSE_cor', self.test_minrmse)]]
            self.epoch_callback_kwargs['namevals'] += [[('Test_RULscore_min', self.test_minscore)]]
            # self.epoch_callback_kwargs['namevals'] += [[('Val_RMSE_cor', self.val_minrmse)]]
            # self.epoch_callback_kwargs['namevals'] += [[('Val_RULscore_min', self.val_minscore)]]
            self.epoch_callback_kwargs['namevals'] += [[('Model_ID', self.model_id)]]

            # need to use lr_scheduler.step, otherwise lr will not be updated
            # self.lr_scheduler.step()
            self.callback_kwargs['epoch_elapse'] = time.time() - epoch_start_time
            # self.callback_kwargs['optimizer_dict'] = optimizer.state_dict()
            self.epoch_end_callback()

        self.logger.info("Optimization done!")


    def train(self, n_user_per_iter):
        assert n_user_per_iter <= self.n_user, f"Can only train {self.n_user} users per step."
        self.metrics.reset()
        sum_sample_inst = 0
        sum_sample_elapse = 0
        sum_update_elapse = 0
        batch_start_time = time.time()
        self.callback_kwargs['prefix'] = 'Train'
        
        chosen_users = random.sample(range(self.n_user), n_user_per_iter)
        for uid in chosen_users:
            # randomly select a user
            self.dataset.reset('train', uid)
            self.load_global_state(uid)
            # print(f'training model of user {uid}')
            
            target_net = self.user_nets[uid]
            target_net.train()
            target_optimizer = self.optimizers[uid]
            target_lr_sche = self.lr_schedulers[uid]
            # i= 0
            for _ in range(self.train_epoch_per_step):
                # for i_batch, (data, data_ops, data_hc, target) in enumerate(self.data_iter):
                for i_batch, dats in enumerate(self.data_iter):
        
                    # i += 1
                    self.callback_kwargs['batch'] = i_batch
                    update_start_time = time.time()

                    # [forward] making next step
                    outputs, losses = self.forward(target_net, dats)

                    # [backward]
                    target_optimizer.zero_grad()
                    for loss in losses: loss.backward()
                    target_optimizer.step()
                    
                    # print(outputs[0].shape)
                    # print(dats[2].shape)
                    # print(losses[0].shape)
                    # exit()
                    # [evaluation] update train metric
                    self.metrics.update([output.data.cpu() for output in outputs], dats[2].cpu(),
                                [loss.data.cpu() for loss in losses])

                    # timing each batch
                    sum_sample_elapse += time.time() - batch_start_time
                    sum_update_elapse += time.time() - update_start_time
                    batch_start_time = time.time()
                    sum_sample_inst += dats[0].shape[0]
                    
            # need to use lr_scheduler.step, otherwise lr will not be updated
            target_lr_sche.step()
        
        # update gloabel model
        self.update_global_net(chosen_users)
            
        # retrive eval results and reset metic
        self.callback_kwargs['namevals'] = self.metrics.get_name_value()
        # speed monitor
        self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
        self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
        # callbacks
        self.step_end_callback()
        self.epoch_callback_kwargs['namevals'] += [[('Train_'+x[0][0],x[0][1])]for x in self.metrics.get_name_value()]

        if self.callback_kwargs['epoch'] == 0:
            self.train_minrmse = self.epoch_callback_kwargs['namevals'][0][0][1]
            self.train_minscore = self.epoch_callback_kwargs['namevals'][1][0][1]
        else:
            # if self.epoch_callback_kwargs['namevals'][0][0][1] < self.train_minrmse:
            #     self.train_minrmse = self.epoch_callback_kwargs['namevals'][0][0][1]

            # if self.epoch_callback_kwargs['namevals'][1][0][1] < self.train_minscore:
            #     self.train_minscore = self.epoch_callback_kwargs['namevals'][1][0][1]

            if self.epoch_callback_kwargs['namevals'][1][0][1] < self.train_minscore:
                self.train_minscore = self.epoch_callback_kwargs['namevals'][1][0][1]
                self.train_minrmse = self.epoch_callback_kwargs['namevals'][0][0][1]    

    # def val(self):

    #     self.metrics.reset()
    #     self.net.eval()
    #     sum_sample_inst = 0
    #     sum_sample_elapse = 0.
    #     sum_update_elapse = 0
    #     batch_start_time = time.time()
    #     self.callback_kwargs['prefix'] = 'Val'

    #     for i_batch, dats in enumerate(self.data_iter):
    #         self.callback_kwargs['batch'] = i_batch
    #         update_start_time = time.time()

    #         # [forward] making next step
    #         outputs, losses = self.forward(dats)

    #         # [evaluation] update train metric
    #         preds = [dats[2][outputs[0][0].argmax(dim=-1).cpu().numpy()]]
    #         mse_loss = torch.pow((preds[0] - dats[2].cpu()), 2).mean()
    #         self.metrics.update(preds, dats[2].cpu(), mse_loss)

    #         # timing each batch
    #         sum_sample_elapse += time.time() - batch_start_time
    #         sum_update_elapse += time.time() - update_start_time
    #         batch_start_time = time.time()
    #         sum_sample_inst += dats[0].shape[0]

    #         if (i_batch % self.step_callback_freq) == 0:
    #             # retrive eval results and reset metic
    #             self.callback_kwargs['namevals'] = self.metrics.get_name_value()
    #             # speed monitor
    #             self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
    #             self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
    #             sum_update_elapse = 0
    #             sum_sample_elapse = 0
    #             sum_sample_inst = 0
    #             # callbacks
    #             self.step_end_callback()

    #     # retrive eval results and reset metic
    #     self.callback_kwargs['namevals'] = self.metrics.get_name_value()
    #     # speed monitor
    #     self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
    #     self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
    #     # callbacks
    #     self.step_end_callback()
    #     self.epoch_callback_kwargs['namevals'] += [[('Val_'+x[0][0],x[0][1])]for x in self.metrics.get_name_value()]

    #     if self.callback_kwargs['epoch'] == 0:
    #         self.val_minrmse = self.epoch_callback_kwargs['namevals'][4][0][1]
    #         self.val_minscore = self.epoch_callback_kwargs['namevals'][5][0][1]
    #     else:
    #         # if self.epoch_callback_kwargs['namevals'][4][0][1] < self.val_minrmse:
    #         #     self.val_minrmse = self.epoch_callback_kwargs['namevals'][4][0][1]

    #         # if self.epoch_callback_kwargs['namevals'][5][0][1] < self.val_minscore:
    #         #     self.val_minscore = self.epoch_callback_kwargs['namevals'][5][0][1]
            
    #         if self.epoch_callback_kwargs['namevals'][5][0][1] < self.val_minscore:
    #             self.val_minscore = self.epoch_callback_kwargs['namevals'][5][0][1]
    #             self.val_minrmse = self.epoch_callback_kwargs['namevals'][4][0][1]

    def test(self):
        self.metrics.reset()
        sum_sample_inst = 0
        sum_sample_elapse = 0.
        sum_update_elapse = 0
        # total_batch = 0
        batch_start_time = time.time()
        self.callback_kwargs['prefix'] = 'Test'
        
        for uid in range(self.n_user):
            self.dataset.reset('test', uid) # ned update
            self.load_global_state(uid)
            # print(f'testing model of user {uid}')
            target_net = self.user_nets[uid]
            target_net.eval()

            for i_batch, dats in enumerate(self.data_iter):
                update_start_time = time.time()
                # [forward] making next step

                outputs, losses = self.forward(target_net, dats)

                # [evaluation] update train metric
                self.metrics.update([output.data.cpu() for output in outputs], dats[2].cpu(),
                           [loss.data.cpu() for loss in losses])

                # timing each batch
                sum_sample_elapse += time.time() - batch_start_time
                sum_update_elapse += time.time() - update_start_time
                batch_start_time = time.time()
                sum_sample_inst += dats[0].shape[0]
                
                # total_batch += 1
                # if (total_batch % self.step_callback_freq) == 0:
                #     # retrive eval results and reset metic
                #     self.callback_kwargs['namevals'] = self.metrics.get_name_value()
                #     # speed monitor
                #     self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
                #     self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
                #     sum_update_elapse = 0
                #     sum_sample_elapse = 0
                #     sum_sample_inst = 0
                #     # callbacks
                #     self.step_end_callback()

        # retrive eval results and reset metic
        self.callback_kwargs['namevals'] = self.metrics.get_name_value()
        # speed monitor
        self.callback_kwargs['sample_elapse'] = sum_sample_elapse / sum_sample_inst
        self.callback_kwargs['update_elapse'] = sum_update_elapse / sum_sample_inst
        # callbacks
        self.step_end_callback()
        self.epoch_callback_kwargs['namevals'] += [[('Test_'+x[0][0],x[0][1])]for x in self.metrics.get_name_value()]

        if self.callback_kwargs['epoch'] == 0:
            self.test_minrmse = self.epoch_callback_kwargs['namevals'][2][0][1]
            self.test_minscore = self.epoch_callback_kwargs['namevals'][3][0][1]
            self.model_id = self.callback_kwargs['epoch'] + 1
        else:
            # if self.epoch_callback_kwargs['namevals'][2][0][1] < self.test_minrmse:
            #     self.test_minrmse = self.epoch_callback_kwargs['namevals'][2][0][1]

            # if self.epoch_callback_kwargs['namevals'][3][0][1] < self.test_minscore:
            #     self.test_minscore = self.epoch_callback_kwargs['namevals'][3][0][1]

            if self.epoch_callback_kwargs['namevals'][3][0][1] < self.test_minscore:
                self.test_minscore = self.epoch_callback_kwargs['namevals'][3][0][1]
                self.test_minrmse = self.epoch_callback_kwargs['namevals'][2][0][1]
                self.model_id = self.callback_kwargs['epoch'] + 1


    def forward(self, net, dats):
        """ typical forward function with:
            dats: data, data_ops, data_hc, target
            single output and single loss
        """
        if net.training:
            torch.set_grad_enabled(True)
            input_var = dats[0].float().cuda()
            ops_var = dats[1].float().cuda()
            target_var = dats[2].float().cuda()
        else:
            torch.set_grad_enabled(False)
            with torch.no_grad():
                input_var = dats[0].float().cuda(non_blocking=True)
                ops_var = dats[1].float().cuda(non_blocking=True)
                target_var = dats[2].float().cuda(non_blocking=True)
            
        output = net(input_var, ops_var)
            
        if hasattr(self, 'criterion') and self.criterion is not None and dats[-1] is not None and target_var is not None:
            loss = self.criterion(output, target_var)
        else:
            loss = None
        return [output], [loss]