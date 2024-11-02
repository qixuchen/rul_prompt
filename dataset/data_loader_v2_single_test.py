"""
Original Author: Ruibing Jin
Based on data_loader_v2_test.py,
change it to only test a single engine for life-time RUL
By Ruibing 2022.1.11
"""
import os
import csv
import random
import numpy as np
from numpy.core.fromnumeric import transpose
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import interpolate
import math

import torch.utils.data as data
import logging
from config import config

class CMPDataIter(data.IterableDataset):

    def __init__(self, data_root, data_set, max_rul, seq_len, net_name, test_id):
        super(CMPDataIter, self).__init__()
        # load params
        self.data_root = data_root
        self.data_set = data_set
        self.max_rul = max_rul
        self.seq_len = seq_len
        self.net_name = net_name
        self.column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                            's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                            's15', 's16', 's17', 's18', 's19', 's20', 's21']
        self.mode = None
        self.val_fold = 0
        self.test_id = test_id

        # load CMAPSS_data
        self.train_data_df, self.test_data_df, self.test_truth = self._get_data(data_root=data_root, data_set=data_set)

        logging.info("CMPDataIter:: iterator initialized (train dataset: '{:s}', shape: {:})".format(data_set, self.train_data_df.shape))
        logging.info("CMPDataIter:: iterator initialized (test dataset: '{:s}', shape: {:})".format(data_set, self.test_data_df.shape))
    
        self.test_x, self.test_ops, self.test_y = self._process(self.train_data_df, self.test_data_df, self.test_truth)

        logging.info("CMPDataIter:: iterator initialized (test data shape: {:})".format(len(self.test_x)))
        logging.info("CMPDataIter:: iterator initialized (test operation shape: {:})".format(len(self.test_ops)))
        logging.info("CMPDataIter:: iterator initialized (test label shape: {:})".format(len(self.test_y)))

        self.initial()
        logging.info("CMPDataIter:: initialize the dataset")

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
    
    def _process(self, train_df, test_df, test_truth):

        # process train data
        train_rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
        train_rul.columns = ['id', 'max']
        train_df = train_df.merge(train_rul, on=['id'], how='left')
        train_y = pd.DataFrame(data=[train_df['max'] - train_df['cycle']]).T

        train_df.drop('max', axis=1, inplace=True)
        train_df.drop(['s1', 's5', 's6', 's10', 's16', 's18', 's19'], axis =1, inplace = True)

        train_df['setting1'] = train_df['setting1'].round(1)
        if self.data_set == "FD003" or self.data_set == "FD001":
            train_df['setting2'] = 0.0
        else:
            train_df['setting2'] = train_df['setting2'].round(3)

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
        if self.data_set == "FD003" or self.data_set == "FD001":
            test_df['setting2'] = 0.0
        else:
            test_df['setting2'] = test_df['setting2'].round(3)

        test_y = test_y.apply(lambda x: [y if y <= self.max_rul else self.max_rul for y in x])
        test_engine_num = test_df['id'].nunique()
        logging.info("CMPDataIter:: iterator initialized (test engine number: {:})".format(test_engine_num))

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

        seq_gen = []
        start_index = 0
        for i in range(test_engine_num):
            end_index = start_index+test_rul.loc[i, 'max']
            if i+1 < self.test_id:
                start_index = end_index
                continue
            else:
                if end_index - start_index < self.seq_len:
                    print('Sensor::test data ({:}) less than seq_len ({:})!'
                        .format(end_index - start_index, self.seq_len))

                    # simply pad the first data serveral times:
                    print('Sensor::Use first data to pad!')
                    num_pad = self.seq_len - (end_index - start_index)
                    new_sg = test_normalized.iloc[start_index:end_index, :]
                    for idx in range(num_pad):
                        new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)

                    val=list(self.gen_sequence(new_sg, self.seq_len, test_normalized.columns))
                else:
                    val=list(self.gen_sequence(test_normalized.iloc[start_index:end_index, :], self.seq_len, test_normalized.columns))
                seq_gen.extend(val)
                start_index = end_index
                test_x = list(seq_gen)
                break

        seq_gen = []
        start_index = 0
        for i in range(test_engine_num):
            end_index = start_index+test_rul.loc[i, 'max']
            if i+1 < self.test_id:
                start_index = end_index
                continue
            else:
                if end_index - start_index < self.seq_len:
                    print('Setting::test data ({:}) less than seq_len ({:})!'
                        .format(end_index - start_index, self.seq_len))

                    # simply pad the first data serveral times:
                    print('Setting::Use first data to pad!')
                    num_pad = self.seq_len - (end_index - start_index)
                    new_sg = test_setting.iloc[start_index:end_index, :]
                    for idx in range(num_pad):
                        new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)
                        
                    val=list(self.gen_sequence(new_sg, self.seq_len, test_setting.columns))
                else:
                    val=list(self.gen_sequence(test_setting.iloc[start_index:end_index, :], self.seq_len, test_setting.columns))
                seq_gen.extend(val)
                start_index = end_index
                test_ops = list(seq_gen)
                break

        seq_gen = []
        start_index = 0
        for i in range(test_engine_num):
            end_index = start_index+test_rul.loc[i, 'max']
            if i+1 < self.test_id:
                start_index = end_index
                continue
            else:
                if end_index - start_index < self.seq_len:
                    print('Setting::test data ({:}) less than seq_len ({:})!'
                        .format(end_index - start_index, self.seq_len))

                    # simply pad the first data serveral times:
                    print('Setting::Use first data to pad!')
                    num_pad = self.seq_len - (end_index - start_index)
                    new_sg = test_y.iloc[start_index:end_index, :]
                    for idx in range(num_pad):
                        new_sg = pd.concat([new_sg.head(1), new_sg], axis=0)
                    val=list([self.gen_labels(new_sg, self.seq_len, test_y.columns)])
                else:
                    val=list(self.gen_labels(test_y.iloc[start_index:end_index, :], self.seq_len, test_y.columns))
                seq_gen.extend(val)
                start_index = end_index
                test_y = list(seq_gen)
                break

        # generate position encoding:
        if 'pe' in self.net_name:
            self.local_pe = self.gen_pe(self.seq_len, 16)

        return test_x, test_ops, test_y
    
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
    
    # function to generate labels
    def gen_test_labels(self, id_df, seq_length, label):
        # For example:
        # [[1]] 
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]

        # For the test labels, only 1 RUL is required per engine which is the last columns on each engine
        return data_matrix[-1,:]
    
    def gen_pe(self, len_seq, p_dim = 16):

        def get_pe(p):
            return [p / np.power(10000, 2 * (hid_j // 2) / p_dim) for hid_j in range(p_dim)]
        
        pe_table = np.array([get_pe(p_i) for p_i in range(len_seq)])
        pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])  # dim 2i
        pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])  # dim 2i+1
    
        return pe_table
    
    def reset(self):

        self.out_x = self.test_x
        self.out_y = self.test_y
        self.out_ops = self.test_ops

        self.start = 0
        self.end = len(self.out_x)
    
    def initial(self):

        self.out_x = self.test_x
        self.out_y = self.test_y
        self.out_ops = self.test_ops

        self.start = 0
        self.end = len(self.out_x)
    
    def __iter__(self):
        # output self.train_x, self.train_ops, self.train_y, self.test_x, self.test_ops, self.test_y  according to self.mode

        out_x = self.out_x[self.start: self.end]
        out_ops = self.out_ops[self.start: self.end]
        out_y = self.out_y[self.start: self.end]

        if 'pe' in self.net_name:
            out_x = [np.concatenate((item, self.local_pe.copy()), axis=1) for item in out_x]

        sum_iter = zip(out_x, out_ops, out_y)

        return iter(sum_iter)

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