import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

dataset = "FD002"
# engine = 224
max = 125.0

for idx in range(258): # For FD004, there are 248 test trajectories; FD002: 258
    engine = idx +1
    res_list = ['../res/{:}_test_engine{:}.pkl'.format(dataset, engine)]
    if not os.path.exists(res_list[0]):
        print('The path: {:} does not exist, so we skip it.'.format(res_list[0]))
        continue

    mov_res = pickle.load(open(res_list[0], 'rb'))

    mov_pred = np.array(mov_res['rul'])*max
    mov_pred = np.clip(mov_pred, 0.0, None)
    gt = np.array(mov_res['gt'])*max

    # extract info and draw
    print ('Starting plot...')

    plt.figure(1)
    plt.suptitle(dataset)
    # ax = plt.gca()
    # ax.set_aspect(0.5)

    x = np.array(range(len(mov_res['rul'])))+1
    plt.plot(x, gt, '-', label='True RUL', linewidth=3)
    plt.plot(x, mov_pred, "o-", ms = 4, label='Our Method')
    plt.ylabel('Remaining useful life')
    plt.xlabel('Cycle')
    plt.legend(numpoints=1,loc='upper right', prop={'size': 13})
    plt.xlim(0, len(mov_res['rul'])+10)
    plt.ylim(-10, 160)

    # plt.show()
    plt.figure(1).savefig('../figs/{:}_RUL_engine{:}_vis.jpg'.format(dataset, engine))
    print('Save the RUL visualization curve plot at {:}'.format('{:}_RUL_engine{:}_vis.jpg'.format(dataset, engine)))
    plt.cla()

