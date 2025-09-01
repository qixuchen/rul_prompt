"""
Author: Qixu Chen
Based on data_loader_prompt.py,
By Qixu 2024.11.04
"""

import os
import csv
import random
import pickle
import numpy as np
from numpy.core.fromnumeric import transpose
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import interpolate
import math

import torch.utils.data as data
import logging
from config import config

import torch.utils.data as data
import logging
from config import config

class CMPDataIterFedNonIIDV2(data.IterableDataset):
    
    def __init__(self, data_root, data_set, llm_name, max_rul, seq_len, net_name, n_user=5, sample_interval=1, iid='non_iid_v2'):
        super(CMPDataIterFedNonIIDV2, self).__init__()
        # load params
        
        self.data_root = data_root
        self.data_set = 'all_FD'
        self.max_rul = max_rul
        self.seq_len = seq_len
        self.net_name = net_name
        self.llm_name = llm_name
        self.n_user = n_user
        self.iid = iid
        self.sample_interval = sample_interval
        
        self.column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']
        self.mode = None
        self.val_fold = 0
        self.cur_user = 0
        
        self.pmpt_1 = self.load_prompt(self.llm_name)

        # # load CMAPSS_data
        self.train_data_dfs, self.test_data_dfs, self.test_truth_dfs = [], [], []
        self.train_x_per_ds, self.train_ops_per_ds, self.train_y_per_ds, self.train_pmpt1_per_ds, self.train_rul_per_ds, self.test_x_per_ds, self.test_ops_per_ds, \
            self.test_y_per_ds, self.test_pmpt1_per_ds, self.test_rul_per_ds = [], [], [], [], [], [], [], [], [], []
        for ds_id, ds in enumerate(['FD001', 'FD002', 'FD003', 'FD004']):    
            train_data, test_data, test_truth = self._get_data(data_root=data_root, data_set=ds)
            self.train_data_dfs.append(train_data.copy())
            self.test_data_dfs.append(test_data.copy())
            self.test_truth_dfs.append(test_truth.copy())
            [l.append(v) for l, v in zip([self.train_x_per_ds, self.train_ops_per_ds, self.train_y_per_ds, self.train_pmpt1_per_ds, self.train_rul_per_ds, self.test_x_per_ds, self.test_ops_per_ds, \
            self.test_y_per_ds, self.test_pmpt1_per_ds, self.test_rul_per_ds], self._process(train_data, test_data, test_truth, ds_id))]
        
        train_split_ruls, test_split_ruls = self.get_split_ruls()
        self.train_id_mapping, self.test_id_mapping = self.get_id_mapping(train_split_ruls, test_split_ruls)
        self.train_x_per_user, self.train_ops_per_user, self.train_y_per_user, self.train_pmpt1_per_user, self.test_x_per_user, self.test_ops_per_user, \
            self.test_y_per_user, self.test_pmpt1_per_user = self.divide_non_iid_v2()
        
        # generate position encoding:
        if 'pe' in self.net_name:
            self.local_pe = self.gen_pe(self.seq_len, 16)
            
        self.initial()
        logging.info("CMPDataIter:: initialize the dataset")
    
    
    def get_split_ruls(self):
        # decides the range of ruls for each user
        train_rul_list, test_rul_list = [], []
        for train_rul in self.train_rul_per_ds:
            train_rul_list.extend(train_rul['max'].to_list())
        train_rul_list = sorted(train_rul_list)
        train_split_ruls = [train_rul_list[0] - 1] # make sure this is smaller than the smallest rul
        for i in range(1, self.n_user):
            train_split_ruls.append(train_rul_list[int(len(train_rul_list) * i / self.n_user)])
        train_split_ruls.append(train_rul_list[-1])
        for i in range(len(train_split_ruls) - 1):
            assert train_split_ruls[i] < train_split_ruls[i+1], "Invalid split values, maybe too many users?"
        
        for test_rul in self.test_rul_per_ds:
            test_rul_list.extend(test_rul['max'].to_list())
        test_rul_list = sorted(test_rul_list)
        test_split_ruls = [test_rul_list[0] - 1] # make sure this is smaller than the smallest rul
        for i in range(1, self.n_user):
            test_split_ruls.append(test_rul_list[int(len(test_rul_list) * i / self.n_user)])
        test_split_ruls.append(test_rul_list[-1])
        for i in range(len(test_split_ruls) - 1):
            assert test_split_ruls[i] < test_split_ruls[i+1], "Invalid split values, maybe too many users?"
            
        return train_split_ruls, test_split_ruls
    
    
    def get_id_mapping(self, train_split_ruls, test_split_ruls):
        # train_id_mapping[i] is a set containing all engine ids that should be assigned to the i-th user
        train_id_mapping = {}
        test_id_mapping = {}
        for i, train_ruls in enumerate(self.train_rul_per_ds): # the i th dataset
            for j in range(self.n_user): # the j th user
                ds_id_pairs = [(i,id) for id in train_ruls[(train_ruls['max'] > train_split_ruls[j]) & (train_ruls['max'] <= train_split_ruls[j+1])]['id'].unique().tolist()]
                for p in ds_id_pairs:
                    train_id_mapping[p] = j
                    
                
        for i, test_ruls in enumerate(self.test_rul_per_ds): # the i th dataset
            for j in range(self.n_user): # the j th user
                ds_id_pairs = [(i,id) for id in test_ruls[(test_ruls['max'] > test_split_ruls[j]) & (test_ruls['max'] <= test_split_ruls[j+1])]['id'].unique().tolist()]
                for p in ds_id_pairs:
                    test_id_mapping[p] = j
                
        return train_id_mapping, test_id_mapping
    
    
    def divide_non_iid_v2(self):
        # divide the training df using engine id in a non iid v2 fashion
            
        train_x_per_user = [[] for _ in range(self.n_user)]
        for train_x in self.train_x_per_ds:
            for item in train_x:
                assigned_user = self.train_id_mapping[tuple(item[:2])]
                train_x_per_user[assigned_user].append(item[2])
            
        train_ops_per_user = [[] for _ in range(self.n_user)]
        for train_ops in self.train_ops_per_ds:
            for item in train_ops:
                assigned_user = self.train_id_mapping[tuple(item[:2])]
                train_ops_per_user[assigned_user].append(item[2])
            
        train_y_per_user = [[] for _ in range(self.n_user)]
        for train_y in self.train_y_per_ds:
            for item in train_y:
                assigned_user = self.train_id_mapping[tuple(item[:2])]
                train_y_per_user[assigned_user].append(item[2])
            
        train_pmpt1_per_user = [[] for _ in range(self.n_user)]
        for train_pmpt1 in self.train_pmpt1_per_ds:
            for item in train_pmpt1:
                assigned_user = self.train_id_mapping[tuple(item[:2])]
                train_pmpt1_per_user[assigned_user].append(item[2])
            
        test_x_per_user = [[] for _ in range(self.n_user)]
        for test_x in self.test_x_per_ds:
            for item in test_x:
                assigned_user = self.test_id_mapping[tuple(item[:2])]
                test_x_per_user[assigned_user].append(item[2])
            
        test_ops_per_user = [[] for _ in range(self.n_user)]
        for test_ops in self.test_ops_per_ds:
            for item in test_ops:
                assigned_user = self.test_id_mapping[tuple(item[:2])]
                test_ops_per_user[assigned_user].append(item[2])
            
        test_y_per_user = [[] for _ in range(self.n_user)]
        for test_y in self.test_y_per_ds:
            for item in test_y:
                assigned_user = self.test_id_mapping[tuple(item[:2])]
                test_y_per_user[assigned_user].append(item[2])
            
        test_pmpt1_per_user = [[] for _ in range(self.n_user)]
        for test_pmpt1 in self.test_pmpt1_per_ds:
            for item in test_pmpt1:
                assigned_user = self.test_id_mapping[tuple(item[:2])]
                test_pmpt1_per_user[assigned_user].append(item[2])

        return train_x_per_user, train_ops_per_user, train_y_per_user, train_pmpt1_per_user, \
            test_x_per_user, test_ops_per_user, test_y_per_user, test_pmpt1_per_user
            
        
    def _get_data(self, data_root, data_set):
        train_data_pt = os.path.join(data_root, 'CMAPSSData',  'train_'+ data_set +'.txt')
        assert os.path.exists(train_data_pt), 'data path does not exist: {:}'.format(train_data_pt)
        
        test_data_pt = os.path.join(data_root, 'CMAPSSData', 'test_'+ data_set +'.txt')
        assert os.path.exists(test_data_pt), 'data path does not exist: {:}'.format(test_data_pt)

        test_truth_pt = os.path.join(data_root, 'CMAPSSData', 'RUL_'+ data_set +'.txt')
        assert os.path.exists(test_truth_pt), 'data path does not exist: {:}'.format(test_truth_pt)

        train_data_df = pd.read_csv(train_data_pt, sep=" ", header=None)
        train_data_df.drop(train_data_df.columns[[26, 27]], axis=1, inplace=True)
        train_data_df.columns = self.column_names
        train_data_df = train_data_df.sort_values(['id','cycle'])

        test_data_df = pd.read_csv(test_data_pt, sep=" ", header=None)
        test_data_df.drop(test_data_df.columns[[26, 27]], axis=1, inplace=True)
        test_data_df.columns = self.column_names
        test_data_df = test_data_df.sort_values(['id','cycle'])

        test_truth = pd.read_csv(test_truth_pt, sep=" ", header=None)
        test_truth.drop(test_truth.columns[[1]], axis=1, inplace=True)

        return train_data_df, test_data_df, test_truth
    
    
    def _process(self, train_df, test_df, test_truth, ds_id):
        # process train data
        train_rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
        train_rul.columns = ['id', 'max']
        train_df = train_df.merge(train_rul, on=['id'], how='left')
        train_y = pd.DataFrame(data=[train_df['max'] - train_df['cycle']]).T

        train_df.drop('max', axis=1, inplace=True)
        train_df.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis =1, inplace = True)

        train_df['setting1'] = train_df['setting1'].round(1)

        train_y = train_y.apply(lambda x: [y if y <= self.max_rul else self.max_rul for y in x])
        train_engine_num = train_df['id'].nunique()
        logging.info("CMPDataIter:: iterator initialized (train engine number: {:})".format(train_engine_num))

        # process test data
        test_rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
        test_rul.columns = ['id', 'max']

        test_truth.columns = ['more']
        test_truth['id'] = test_truth.index + 1
        test_truth['max'] = test_rul['max'] + test_truth['more']
        test_truth.drop('more', axis=1, inplace=True)

        test_df = test_df.merge(test_truth, on=['id'], how='left')
        test_y = pd.DataFrame(data=[test_df['max'] - test_df['cycle']]).T

        test_df.drop('max', axis=1, inplace=True)
        test_df.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis =1, inplace = True)

        test_df['setting1'] = test_df['setting1'].round(1)

        test_y = test_y.apply(lambda x: [y if y <= self.max_rul else self.max_rul for y in x])
        test_engine_num = test_df['id'].nunique()
        logging.info("CMPDataIter:: iterator initialized (test engine number: {:})".format(test_engine_num))

        if self.sample_interval > 1:
            train_y = train_y.apply(lambda x: x - x % self.sample_interval)
            test_y = test_y.apply(lambda x: x - x % self.sample_interval)
        
        # normailize both train and test data
        train_data = train_df.iloc[:, 2:]
        test_data = test_df.iloc[:, 2:]

        train_normalized = pd.DataFrame(columns = train_data.columns[3:])
        test_normalized = pd.DataFrame(columns = test_data.columns[3:])

        scaler = MinMaxScaler() 

        grouped_train = train_data.groupby('setting1')
        grouped_test = test_data.groupby('setting1')

        for train_idx, train in grouped_train:

            scaled_train = scaler.fit_transform(train.iloc[:, 3:])
            scaled_train_combine = pd.DataFrame(
                    data=scaled_train,
                    index=  train.index,  
                    columns=train_data.columns[3:])
            train_normalized = pd.concat([train_normalized, scaled_train_combine])

            for test_idx, test in grouped_test:
                if train_idx == test_idx:
                    scaled_test = scaler.transform(test.iloc[:, 3:])
                    scaled_test_combine = pd.DataFrame(
                            data=scaled_test,    
                            index=  test.index,  
                            columns=test_data.columns[3:])
                    test_normalized = pd.concat([test_normalized, scaled_test_combine])

        train_normalized = train_normalized.sort_index()
        test_normalized = test_normalized.sort_index()

        # diff@xuqing
        train_setting = scaler.fit_transform(train_df.iloc[:, 1:5])
        test_setting = scaler.transform(test_df.iloc[:, 1:5])

        train_setting = pd.DataFrame(
                        data= train_setting,    
                        index= train_df.index,  
                        columns= train_df.columns[1:5])
        
        test_setting = pd.DataFrame(
                            data=test_setting,    
                            index= test_df.index,  
                            columns=test_df.columns[1:5])
        
        train_y =  train_y.apply(lambda x: (x/self.max_rul)) # norm_y = y/max_RUL
        test_y =  test_y.apply(lambda x: (x/self.max_rul)) # norm_y = y/max_RUL

        condition_num = train_df['setting1'].nunique()

        if condition_num > 1:
            logging.info("CMPDataIter:: data includes multi operating conditions")
        else:
            logging.info("CMPDataIter:: data includes single operating conditions")
        
        # generate final data:
        #generate 9 x 15 windows to obtain train_x
        seq_gen = []
        start_index = 0
        for i in range(train_engine_num):
            end_index = start_index+train_rul.loc[i, 'max']
            if end_index - start_index < self.seq_len-1:
                print('train data less than seq_len!')
            # for sensor train matrix, number of 21 X 15 needed per data points (minus the first sequence length) per engine, so the array input start from start index
            # need to i+1 since loc[i] contains engine with id i+1
            val=[[ds_id, i+1, x] for x in self.gen_sequence(train_normalized.iloc[start_index:end_index, :], self.seq_len, train_normalized.columns)]
            seq_gen.extend(val)
            start_index = end_index
        train_x = seq_gen

        #generate 3 x 15 windows to obtain train_ops
        seq_gen = []
        start_index = 0
        for i in range(train_engine_num):
            end_index = start_index+train_rul.loc[i, 'max']
            if end_index - start_index < self.seq_len-1:
                print('train ops less than seq_len!')
            # for ops train matrix, number of 3 X 15 needed per data points (minus the first sequence length) per engine, so the array input start from start index
            #settings data are in the first 3 columns of Train_Norm
            val=[[ds_id, i+1, x] for x in self.gen_sequence(train_setting.iloc[start_index:end_index, :], self.seq_len, train_setting.columns)]
            seq_gen.extend(val)
            start_index = end_index
        train_ops = seq_gen

        # generate train labels
        seq_gen = []
        start_index = 0
        for i in range(train_engine_num):
            end_index = start_index+train_rul.loc[i, 'max']
            val=[[ds_id, i+1, x] for x in self.gen_labels(train_y.iloc[start_index:end_index, :], self.seq_len, train_y.columns)]
            seq_gen.extend(val)
            start_index = end_index
        train_y = seq_gen

        seq_gen = []
        start_index = 0
        for i in range(test_engine_num):
            end_index = start_index+test_rul.loc[i, 'max']
            #diff@xuqing
            # for test matrix, only 1 of n X 15 needed per engine, so the array input start from end index - sequence length
            if end_index - start_index < self.seq_len:
                print('Sensor::test data ({:}) less than seq_len ({:})!'
                    .format(end_index - start_index, self.seq_len))

                # simply pad the first data serveral times:
                print('Sensor::Use first data to pad!')
                num_pad = self.seq_len - (end_index - start_index)
                new_sg = test_normalized.iloc[start_index:end_index, :]
                for idx in range(num_pad):
                    new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)

                val=[[ds_id, i+1, x] for x in self.gen_sequence(new_sg, self.seq_len, test_normalized.columns)]
            else:
                val=[[ds_id, i+1, x] for x in self.gen_sequence(test_normalized.iloc[end_index - self.seq_len:end_index, :], self.seq_len, test_normalized.columns)]
            seq_gen.extend(val)
            start_index = end_index
        test_x = seq_gen

        seq_gen = []
        start_index = 0
        for i in range(test_engine_num):
            end_index = start_index+test_rul.loc[i, 'max']
            # for test matrix, only 1 of n X 15 needed per engine, so the array input start from end index - sequence length
            if end_index - start_index < self.seq_len:
                print('Setting::test data ({:}) less than seq_len ({:})!'
                    .format(end_index - start_index, self.seq_len))

                # simply pad the first data serveral times:
                print('Setting::Use first data to pad!')
                num_pad = self.seq_len - (end_index - start_index)
                new_sg = test_setting.iloc[start_index:end_index, :]
                for idx in range(num_pad):
                    new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)
                    
                val=[[ds_id, i+1, x] for x in self.gen_sequence(new_sg, self.seq_len, test_setting.columns)]
            else:
                val=[[ds_id, i+1, x] for x in self.gen_sequence(test_setting.iloc[end_index - self.seq_len:end_index, :], self.seq_len, test_setting.columns)]

            seq_gen.extend(val)
            start_index = end_index
        test_ops = seq_gen

        seq_gen = []
        start_index = 0
        for i in range(test_engine_num):
            end_index = start_index+test_rul.loc[i, 'max']
            val=[[ds_id, i+1, np.array([x])] for x in self.gen_test_labels(test_y.iloc[end_index - self.seq_len:end_index, :], test_y.columns)]
            seq_gen.extend(val)
            start_index = end_index
        test_y = seq_gen
        train_pmpt1 = self.generate_prmpts(train_y)
        test_pmpt1 = self.generate_prmpts(test_y)
        
        return train_x, train_ops, train_y, train_pmpt1, train_rul, test_x, test_ops, test_y, test_pmpt1, test_rul
    

    def load_prompt(self, llm_name):
        pmpt_path = './feats'
        assert llm_name in ['clip', 'siglip']
        if llm_name == 'clip':
            pmpt_pt1 = os.path.join('./feats', 'clip_feature_ts_forcasting.pkl')
        else:
            pmpt_pt1 = os.path.join('./feats', 'siglip.pkl')
        return pickle.load(open(pmpt_pt1, 'rb'))
    
    
    def generate_prmpts(self, labels):
        return [[x[0], x[1], self.pmpt_1[int(x[2][0] * self.max_rul)]] for x in labels]


    def gen_sequence(self, id_df, seq_length, seq_cols):
        # for one id I put all the rows in a single matrix
        data_matrix = id_df[seq_cols].values.astype(np.float32)
        num_elements = data_matrix.shape[0]
        # Iterate over two lists in parallel.
        # For example id1 (engine 1) have 192 rows and sequence_length is equal to 15
        # so zip iterate over two following list of numbers (0,177),(14,191)
        # 0 14 -> from row 0 to row 14
        # 1 15 -> from row 1 to row 15
        # 2 16 -> from row 2 to row 16
        # ...
        # 177 191 -> from row 177 to 191
        for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
            yield data_matrix[start:stop, :]
            
            
    def gen_labels(self, id_df, seq_length, label):
        # For example:
        # [[1]
        # [4]
        # [1]
        # [5]
        # [9]
        # ...
        # [200]] 
        data_matrix = id_df[label].values 
        num_elements = data_matrix.shape[0]
        label_matrix = []
        for i in range(num_elements-(seq_length-1)):
            label_matrix.append(data_matrix[i+(seq_length-1), :])

        return label_matrix
    
    
    def gen_test_labels(self, id_df, label):
        # For example:
        # [[1]] 
        data_matrix = id_df[label].values
        # For the test labels, only 1 RUL is required per engine which is the last columns on each engine
        return data_matrix[-1,:]
    
    
    def gen_pe(self, len_seq, p_dim = 16):

        def get_pe(p):
            return [p / np.power(10000, 2 * (hid_j // 2) / p_dim) for hid_j in range(p_dim)]
        
        pe_table = np.array([get_pe(p_i) for p_i in range(len_seq)])
        pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])  # dim 2i
        pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])  # dim 2i+1
    
        return pe_table
        
    
    def reset(self, mode, uid=0):
        assert uid < self.n_user, f"user id should be within [0, {self.n_user}]"
        self.cur_user = uid
        
        if mode == 'train':
            self.mode = 'train'
            self.out_x = self.train_x_per_user[uid]
            self.out_ops = self.train_ops_per_user[uid]
            self.out_y = self.train_y_per_user[uid]
            self.out_prompt1 = self.train_pmpt1_per_user[uid]
            self.start = 0
            self.end = len(self.out_x)
            
        elif mode == 'test':
            self.mode == 'test'
            self.out_x = self.test_x_per_user[uid]
            self.out_ops = self.test_ops_per_user[uid]
            self.out_y = self.test_y_per_user[uid]
            self.out_prompt1 = self.test_pmpt1_per_user[uid]
            self.start = 0
            self.end = len(self.out_x)
    
    
    def initial(self):
        self.mode = 'train'
        self.cur_user = 0
        self.out_x = self.train_x_per_user[self.cur_user]
        self.out_ops = self.train_ops_per_user[self.cur_user]
        self.out_y = self.train_y_per_user[self.cur_user]
        self.out_prompt1 = self.train_pmpt1_per_user[self.cur_user]
        self.start = 0
        self.end = len(self.out_x)
    
    
    # def cross_fold(self, data_list):
    #     ref_data = data_list[0]
    #     num_data = len(ref_data)
    #     group_size = num_data // 5

    #     zip_list = list(zip(data_list[0], data_list[1], data_list[2], data_list[3]))
    #     random.shuffle(zip_list)
    #     train_x, train_ops, train_y, train_pmpt1 = zip(*zip_list)

    #     grouped_train_x = []
    #     grouped_train_ops = []
    #     grouped_train_y = []
    #     grouped_prmpt_1 = []


    #     for g_id in range(4):
    #         group_train_x = train_x[0+g_id*group_size:(g_id+1)*group_size]
    #         group_train_ops = train_ops[0+g_id*group_size:(g_id+1)*group_size]
    #         group_train_y = train_y[0+g_id*group_size:(g_id+1)*group_size]
    #         group_prmpt_1 = train_pmpt1[0+g_id*group_size:(g_id+1)*group_size]

    #         grouped_train_x.append(group_train_x)
    #         grouped_train_ops.append(group_train_ops)
    #         grouped_train_y.append(group_train_y)
    #         grouped_prmpt_1.append(group_prmpt_1)

        
    #     grouped_train_x.append(train_x[4*group_size:])
    #     grouped_train_ops.append(train_ops[4*group_size:])
    #     grouped_train_y.append(train_y[4*group_size:])
    #     grouped_prmpt_1.append(train_pmpt1[4*group_size:])

    #     return grouped_train_x, grouped_train_ops, grouped_train_y, grouped_prmpt_1


    def __iter__(self):
        # output self.train_x, self.train_ops, self.train_y, self.train_prompt1,
        # or self.test_x, self.test_ops, self.test_y, self.test_prompt1 according to self.mode
        
        out_x = self.out_x[self.start: self.end]
        out_ops = self.out_ops[self.start: self.end]
        out_y = self.out_y[self.start: self.end]
        out_prompt1 = self.out_prompt1[self.start: self.end]
        
        if 'pe' in self.net_name:
            out_x = [np.concatenate((item, self.local_pe.copy()), axis=1) for item in out_x]
        
        return iter(zip(out_x, out_ops, out_y, out_prompt1))


    def __len__(self):
        return len(self.out_x)
    
def worker_init_fn(worker_id):
    worker_info = data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)