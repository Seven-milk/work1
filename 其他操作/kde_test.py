# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

x = np.random.normal(loc=0, scale=0.3, size=1000)
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.hist(x)
# ax.plot(x, np.zeros(x.shape), 'b+', ms=20)
kde = stats.gaussian_kde(x, bw_method="scott")
# kde2 = stats.gaussian_kde(x, bw_method="silverman")
# x_ = np.linspace(0, 1, num=200)
# ax2.plot(x_, kde1(x_), 'k-', label="scott")
# ax2.plot(x_, kde2(x_), 'r-', label="silverman")
# x.sort()
# ax.plot(x, [kde1.integrate_box_1d(low=0, high=x[i]) for i in range(len(x))], 'k-', label="scott")
# ax.plot(x, [kde2.integrate_box_1d(low=0, high=x[i]) for i in range(len(x))], 'k-', label="silverman")
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.hist(x, label="Hist", alpha=0.5)  #, bins=int(kde.neff)
x_eval = np.linspace(x.min(), x.max(), num=int((x.max() - x.min())*100))
ax1.plot(x, np.zeros(x.shape), '+', color='navy', ms=20, label="Samples")
ax1.set_ylabel("Number of samples")
ax2.plot(x_eval, kde(x_eval), 'r-', label="KDE based on bw_method: " + "scott")  # pdf
ax2.set_ylabel("PDF")
ax1.set_title("Kernel density estimation")
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

# x_percentile vs x
x_percentile = np.array([kde.integrate_box_1d(low=0, high=x[i]) for i in range(len(x))])
fig1 = plt.figure()
# plt.plot(list(range(len(x))), x, 'r+')
plt.plot(list(range(len(x))), x, 'r')
plt.plot(list(range(len(x))), x_percentile, 'b')
