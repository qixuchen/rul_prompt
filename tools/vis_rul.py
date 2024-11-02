import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

dataset = "FD004"
max = 125.0
res_list = ['res/{:}_test.pkl'.format(dataset), '{:}_test.pkl'.format(dataset)]

mov_res = pickle.load(open(res_list[0], 'rb'))
base_res = pickle.load(open(res_list[1], 'rb'))

mov_pred = np.array(mov_res['rul'])*max
base_pred = np.array(base_res['rul'])*max
mov_pred = np.clip(mov_pred, 0.0, None)
base_pred = np.clip(base_pred, 0.0, None)
gt = np.array(mov_res['gt'])*max

# sort based on RUL
gt_arg = np.argsort(gt)
mov_pred = mov_pred[gt_arg]
base_pred = base_pred[gt_arg]
gt = gt[gt_arg]

# extract info and draw
print ('Starting plot...')
plt.style.use('seaborn-whitegrid')

plt.figure(figsize = (10,3))
plt.suptitle(dataset)
ax = plt.gca()
ax.set_aspect(0.4)

assert len(mov_res['rul']) == len(base_res['rul']), 'mov data should be equal to base data'

x = np.array(range(len(mov_res['rul'])))+1
plt.plot(x, mov_pred, "o", label='Our Method')
# plt.plot(x, base_pred, '*', label='Baseline')
plt.plot(x, gt, 'g-', label='True RUL', linewidth=3)
plt.ylabel('Remaining useful life')
plt.xlabel('Test unit with increasing RUL')
plt.legend(numpoints=1,loc='upper left', prop={'size': 13})
plt.ylim(-10, 160)

# plt.show()
plt.figure(1).savefig('figs/{:}_RUL_vis.jpg'.format(dataset))
print('Save the RUL visualization curve plot at {:}'.format( dataset + '_RUL_vis.jpg'))

