# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

import numpy as np
import pandas as pd
import time


# P = np.loadtxt('F:/小论文2/代码/results/daily_gldas/Rainf_f_tavg_array_daily.txt')
# x = P[:, 1]
# x = np.random.randn(10000)
# x = pd.read_excel('F:/小论文2/代码/results/P_test.xlsx').values.flatten()
# x[0] = 0.1
# x[1] = 0.1
# x[-1] = 0.1
# x[-2] = 0.1
# h = 0.1

# starttime1 = time.time()
# flag_start = []
# flag_end = []
# 判断第一个点是否为start，当第一个为start时判断第一个是否为end（基于第二个）
# if x[0] <= h:
#     flag_start.append(0)
#     if x[1] > h:
#         flag_end.append(0)
# # 中间点判断,要保证内部都小于h
# for i in range(1, len(x) - 1):
#     if (x[i] <= h) & (x[i - 1] > h):
#         flag_start.append(i)
#     if (x[i] <= h) & (x[i + 1] > h):
#         flag_end.append(i)
# 判断最后一个点是否为end， 当为end时判断是否为开始(基于倒数第二个)
# if x[len(x) - 1] <= h:
#     flag_end.append(len(x) - 1)
#     if x[len(x) - 2] > h:
#         flag_start.append(len(x) - 1)
# # 转换为数组
# flag_start = np.array(flag_start)
# flag_end = np.array(flag_end)
# endtime1 = time.time()
# dtime1 = endtime1 - starttime1
# print(dtime1)


# starttime2 = time.time()
# dry_flag = np.argwhere(x <= h).flatten()
# flag_start2 = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()[1:]
# flag_end2 = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()[:-1]
#
#
# if x[dry_flag[0]] <= h:
#     flag_start2 = np.insert(flag_start2, 0, dry_flag[0])
    # if x[dry_flag[1]] > h:
    #     flag_end2 = np.insert(flag_end2, 0, dry_flag[0])

# if x[dry_flag[0]] == h and x[dry_flag[1]] < h:
#     flag_start2 = np.insert(flag_start2, 0, dry_flag[0])


# if x[dry_flag[-1]] <= h:
#     flag_end2 = np.append(flag_end2, dry_flag[-1])
    # if x[dry_flag[-2]] > h:
    #     flag_start2 = np.append(flag_start2, dry_flag[-1])
# if x[dry_flag[-1]] == h and x[dry_flag[-2]] < h:
#     flag_end2 = np.append(flag_end2, dry_flag[-1])




# endtime2 = time.time()
# dtime2 = endtime2 - starttime2
# print(dtime2)
# flag_start1 = np.array(flag_start1)
# flag_end1 = np.array(flag_end1)
#
# dry_flag = np.argwhere(x <= 1).flatten()
# flag_start2 = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()
# np.argwhere(flag_start1 - flag_start2[1:] != 0)
#
# dry_flag = np.argwhere(x <= 1).flatten()
# flag_end2 = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()
# np.argwhere(flag_end1 - flag_end2[:-1] != 0)
# starttime2 = time.time()
# dry_flag = np.argwhere(x <= 1).flatten()
# flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()
# flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()
# endtime2 = time.time()
# dtime2 = endtime2 - starttime2
# print(dtime2)
# x = abs(x - h)
# starttime1 = time.time()
# s1 = np.array([x[flag_start2[i]:flag_end2[i] + 1].sum() for i in range(len(flag_start2))])
# endtime1 = time.time()
# dtime1 = endtime1 - starttime1
#
#
# starttime2 = time.time()
# s2 = np.zeros(len(flag_start2))
# for i in range(len(s2)):
#     s2[i] = x[flag_start2[i]:flag_end2[i] + 1].sum()
# endtime2 = time.time()
# dtime2 = endtime2 - starttime2


# starttime3 = time.time()
# s3 = map(lambda x : x[flag_start2[i]:flag_end2[i] + 1].sum(), x)
# endtime3 = time.time()
# dtime3 = endtime3 - starttime3



# print('t1:{} \n t2:{} \n t3:{}'.format(dtime1, dtime2, dtime3))
# result = np.argwhere(s1!=s3)
# D = flag_end2 - flag_start2 + 1
# x = abs(x - h)  # 烈度 = x - h 的绝对值
# ------------------------------速度慢，用列表推导替代
# for i in range(len(S)):
#     S[i] = x[flag_start[i]:flag_end[i] + 1].sum()
# ---------------------------------
# S = np.array([x[flag_start2[i]:flag_end2[i] + 1].sum() for i in range(len(flag_start2))])
