import matplotlib.pyplot as plt
import matplotlib

def autolabel(rects, ax):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.01*h, h,
                ha='center', va='bottom',fontsize=15)

y = [1802.40, 1626.69, 1194.35, 1103.18, 927.43, 974.28, 843.94, 871.34]
mse = [20.59, 19.62, 16.41, 15.40, 15.04, 15.29, 14.33, 14.45]
wid = 0.1
x = []
for idx in range(len(mse)):
    x.extend([wid*1.5*idx])

cl_dict = {0: 'chartreuse', 1: 'lime', 2: 'lightgreen', 3: 'cyan', 4: 'darkturquoise',\
            5: 'deepskyblue', 6: 'dodgerblue', 7: 'royalblue'}

fig,(ax1,ax2)= plt.subplots(1,2)
 
rects1 = ax1.bar(x[0], mse[0], width= wid, color = cl_dict[7])
rects2 = ax1.bar(x[1], mse[1], width= wid, color = cl_dict[6])
rects3 = ax1.bar(x[2], mse[2], width= wid, color = cl_dict[5])
rects4 = ax1.bar(x[3], mse[3], width= wid, color = cl_dict[4])
rects5 = ax1.bar(x[4], mse[4], width= wid, color = cl_dict[2])
rects6 = ax1.bar(x[5], mse[5], width= wid, color = cl_dict[3])
rects7 = ax1.bar(x[6], mse[6], width= wid, color = cl_dict[0])
rects8 = ax1.bar(x[7], mse[7], width= wid, color = cl_dict[1])
ax1.plot(x, mse, linewidth = 5, marker = 'o', ms = 15, mfc = 'orange', mec = 'orange', color = 'orange')

ax1.set_title('RMSE',fontsize=16)
ax1.set_ylim(min(mse)-0.5, max(mse)+0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(['15', '20', '25', '30', '35', '40', '45', '50'], Fontsize = 16)
ax1.tick_params(axis='y', labelsize=16)
# ax1.legend( (rects1, rects2, rects3), ('16 bit', '32 bit', '64 bit'), loc='upper right', fontsize=12)
# plt.xlabel('Window Length',fontsize=16)

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax1)
autolabel(rects4, ax1)
autolabel(rects5, ax1)
autolabel(rects6, ax1)
autolabel(rects7, ax1)
autolabel(rects8, ax1)

rects1 = ax2.bar(x[0], y[0], width= wid, color = cl_dict[7])
rects2 = ax2.bar(x[1], y[1], width= wid, color = cl_dict[6])
rects3 = ax2.bar(x[2], y[2], width= wid, color = cl_dict[5])
rects4 = ax2.bar(x[3], y[3], width= wid, color = cl_dict[4])
rects5 = ax2.bar(x[4], y[4], width= wid, color = cl_dict[2])
rects6 = ax2.bar(x[5], y[5], width= wid, color = cl_dict[3])
rects7 = ax2.bar(x[6], y[6], width= wid, color = cl_dict[0])
rects8 = ax2.bar(x[7], y[7], width= wid, color = cl_dict[1])
ax2.plot(x, y, linewidth = 5, marker = 'o', ms = 15, mfc = 'orange', mec = 'orange', color = 'orange')

ax2.set_title('Score',fontsize=16)
ax2.set_ylim(min(y)-50, max(y)+70)
ax2.set_xticks(x)
ax2.set_xticklabels(['15', '20', '25', '30', '35', '40', '45', '50'], Fontsize = 16)
ax2.tick_params(axis='y', labelsize=16)
# ax2.legend( (rects1, rects2, rects3), ('16 bit', '32 bit', '64 bit'), loc='upper right', fontsize=12)
# plt.xlabel('Postion Encoding Dimension',fontsize=16)
# fig.text(0.5, 0.01, 'Position Encoding Dimension', ha='center', fontsize=12)
# fig.xlabel('Position Encoding Dimension')

autolabel(rects1, ax2)
autolabel(rects2, ax2)
autolabel(rects3, ax2)
autolabel(rects4, ax2)
autolabel(rects5, ax2)
autolabel(rects6, ax2)
autolabel(rects7, ax2)
autolabel(rects8, ax2)

plt.tight_layout()
plt.show()