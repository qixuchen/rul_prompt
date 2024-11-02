import os
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

dataset = 'FD004'
config_name = 'clip'
exp_time = '2023-09-15-14-23'
cur_pt = os.path.dirname(os.path.abspath(__file__))
exp_names = 'experiment_{:}_{:}.log'.format(config_name, exp_time)
log_pt = os.path.join(cur_pt , 'output', config_name, dataset, exp_names)

main_key = 'Final_Epoch'
train_keys = ['Train_RMSE', 'Train_RULscore']
val_keys = ['Val_RMSE', 'Val_RULscore']
test_keys = ['Test_RMSE', 'Test_RULscore']

print ('Loading the log file experiment_{:}_{:}.log'.format(config_name, exp_time))

total_lines = []

with open(log_pt) as f:
    total_lines = [x.strip() for x in f.readlines()]
f.close

# filter redundant lines
scalar_np = np.array(total_lines)
keep_inds = [main_key in x for x in scalar_np]
loss_inds = np.argwhere(keep_inds)[:, 0] +1
loss_lines = scalar_np[loss_inds]
# loss_lines = loss_lines[:50]

# extract info and draw
print ('Starting plot...')
plt.style.use('seaborn-whitegrid')

plt.figure(1)
plt.suptitle(exp_names + 'BASIC')

for ind in range(len(train_keys)):
    key_name1 = train_keys[ind]
    key_name2 = val_keys[ind]
    key_name3 = test_keys[ind]
    # info_list1 = [np.around(float(re.findall(r"-?\d+\.\d*", re.findall(r"%s\W+\d+\.\d+" % (key_name1), x)[0])[0]), 2) for x in loss_lines]
    if ind == 1: 
        info_list1 = [np.around(float(re.findall(r"-?\d+\.\d*", re.findall(r"%s\W+\d+\.\d+" % (key_name1), x)[0])[0]), 2)/1735.8 for x in loss_lines]
        info_list2 = [np.around(float(re.findall(r"-?\d+\.\d*", re.findall(r"%s\W+\d+\.\d+" % (key_name2), x)[0])[0]), 2)/43.40 for x in loss_lines]
    else:
        info_list1 = [np.around(float(re.findall(r"-?\d+\.\d*", re.findall(r"%s\W+\d+\.\d+" % (key_name1), x)[0])[0]), 2) for x in loss_lines]
        info_list2 = [np.around(float(re.findall(r"-?\d+\.\d*", re.findall(r"%s\W+\d+\.\d+" % (key_name2), x)[0])[0]), 2) for x in loss_lines]

    info_list3 = [np.around(float(re.findall(r"-?\d+\.\d*", re.findall(r"%s\W+\d+\.\d+" % (key_name3), x)[0])[0]), 2) for x in loss_lines]

    assert len(info_list1) == len(info_list2), 'train data should be equal to val data'

    plt.subplot(1, 2, ind+1)
    x = np.array(range(len(info_list1)-1))+1
    plt.plot(x, info_list1[1:], '-', label='train')
    plt.plot(x, info_list2[1:], '-', label='val')
    plt.plot(x, info_list3[1:], '-', label='test')
    plt.ylabel(key_name1[6:])
    plt.xlabel('Epoch')
    plt.legend()

    # max_y = max(max(max(info_list1, info_list2), info_list3))
    # max_y = max(max(info_list3, info_list2))
    # min_y = min(min(min(info_list1, info_list2), info_list3))
    # max_y = max(info_list1)
    # min_y = min(info_list1)
    # num_yticks = int((max_y - min_y) / 0.005)
    # plt.yticks = np.linspace(min_y, max_y, num_yticks)

plt.show()
# plt.figure(1).savefig('{:}_exp_curve.jpg'.format(dataset + '_' + exp_names[:-4]))
# print('Save the loss curve plot at {:}'.format(dataset + '_' + exp_names + '_exp_curve.jpg'))

