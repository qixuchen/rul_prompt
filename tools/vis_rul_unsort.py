import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

dataset = "FD002"
max = 125.0
fig_name = '../figs/{:}_adanet_RUL_vis.jpg'.format(dataset)
res_list = '../res/{:}_adanet_test.pkl'.format(dataset)

res = pickle.load(open(res_list, 'rb'))
pred = np.array(res['rul'])*max
pred = np.clip(pred, 0.0, None)
gt = np.array(res['gt'])*max

# choose the pre-defined figure setting
if (dataset == "FD001") or dataset == ("FD003"):
    fig_size = (8,4)
    fig_aspect = 0.22
    fig_xlocator = 20
    fig_xlim = 100
    
else:
    fig_size = (8,3)
    fig_aspect = 0.55
    fig_xlocator = 50 
    fig_xlim = 250
  

# extract info and draw
print ('Starting plot...')

plt.figure(1, figsize = fig_size, dpi=300)
plt.suptitle(dataset)
ax = plt.gca()
ax.set_aspect(fig_aspect)

x = np.array(range(len(res['rul'])))
# plt.plot(x, gt, "--", color = 'coral', label='True RUL')
# plt.plot(x, gt, 'g-', label='True RUL', linewidth=3)
plt.plot(x, gt, "b-", linewidth=0.5, label='True RUL')
plt.plot(x, pred, 'rs--', ms=1, linewidth=0.5, label='Our Method')

plt.ylabel('RUL', fontdict={'weight': 'bold', 'size':12})
plt.xlabel('Test Engine ID', fontdict={'weight': 'bold', 'size':12})
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.legend(ncol=2, numpoints=1,loc='upper left', prop={'size': 10})

ax.xaxis.set_major_locator(plt.MultipleLocator(fig_xlocator))
ax.yaxis.set_major_locator(plt.MultipleLocator(25))
plt.xlim(0, fig_xlim)
plt.ylim(0, 130)


# plt.show()
plt.figure(1).savefig(fig_name, dpi=300)
print('Save the RUL visualization curve plot at {:}'.format(fig_name))

