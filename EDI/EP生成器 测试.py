# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

import numpy
import time

# P = numpy.random.randn(10000)
P = numpy.arange(10000)
DEP = P
# DS = 20
DS = numpy.zeros(len(P) - 20 + 1)
DS = DS.astype('int')
DS[:] = 50
DS[3] = 21
DS[4:8] = [22,23,24,25]

#  -----------------------------------------------------
# wight = 0
# wight_array = numpy.zeros(len(DEP))  # 去尾，为了与DEP对应，才能相除
# # (len(DEP) - DS + 1) - (len(DEP) - DS + 1) % 365
# for i in range(DS):
#     wight += 1 / (i + 1)
# wight_array[:] = wight
#
# wight2 = sum([1/(i+1) for i in range(DS)])
starttime1 = time.time()
wight_array = numpy.zeros(len(DEP))
for i in range(len(DS)):
    wight = 0
    for j in range(DS[i]):
        wight += 1 / (j + 1)
    wight_array[i] = wight

endtime1 = time.time()
dtime1 = endtime1 - starttime1
print(dtime1)

starttime2 = time.time()
wight_array2 = numpy.zeros(len(DEP))
for i in range(len(DS)):
    wight_array2[i] = sum([1/(j+1) for j in range(DS[i])])
endtime2 = time.time()
dtime2 = endtime2 - starttime2
print(dtime1)
# p_sum1 = numpy.array([P[i + DS[0] - DS[i]:i + DS[0]].sum() for i in range(len(P) - DS[0] + 1)])
# p_sum2 = numpy.zeros(len(P) - DS[0] + 1)
# for i in range(len(P) - DS[0] + 1):
#     p_sum2[i] = P[i + DS[0] - DS[i]:i + DS[0]].sum()
# def EP0(P, DS):
#     '''定义生成器计算EP'''
#     EP = np.zeros(len(P) - DS + 1)
#     for i in range(len(P) - DS + 1):
#         s = 0
#         for j in range(DS):
#             s += P[j+i: DS+i].mean()
#         EP[i] = s
#     return EP




# def EP1(P, DS):
#     '''原生代码计算EP'''
#     EP = np.zeros(len(P) - DS + 1)
#     for i in range(len(P) - DS + 1):
#         s_ = 0
#         for j in range(1, DS):
#             s_ += P[i + DS - 1 - j:i + DS].mean()
#         EP[i] = s_ + P[i + DS - 1]
#     return EP
#
#
# starttime1 = time.time()
# EP_0 = EP0(P,DS)
# endtime1 = time.time()
# dtime1 = endtime1 - starttime1
# print(dtime1)
# #
# starttime2 = time.time()
# EP_1 = EP1(P,DS)
# endtime2 = time.time()
# dtime2 = endtime2 - starttime2
# print(dtime2)
#
#
# result=(EP_0 == EP_1)
# print(np.argwhere(result==False))

# def EP_generator(P, DS, DS0, i=0):
#     '''定义生成器计算EP'''
#     while True:
#         s = 0
#         for j in range(DS):
#             # 为了让变DS可能，这个DS_range要变化而下方的DS不能变，每次更改DS而不更改DS0
#             s += P[DS0 + i - j - 1: DS0 + i].mean()
#         [i, DS] = yield s



# EP_2_ = EP_generator(P, DS, DS)
# EP_2 = numpy.zeros(len(P) - DS + 1)
# EP_2[0] = next(EP_2_)
# for i in range(1, len(P) - DS + 1):
#     EP_2[i] = EP_2_.send([i, DS])
# def fun_EP2(P, DS=365):
#     '''使用生成器计算EP'''
#     if type(DS) is numpy.ndarray:
#         EP = numpy.zeros(len(P) - DS[0] + 1)
#         EP_ = EP_generator(P, DS[0], DS[0])
#         EP[0] = next(EP_)
#         for i in range(1, len(P) - DS[0] + 1):
#             EP[i] = EP_.send([i, DS[i]])
#         return EP
#         EP_.close()
#     else:
#         EP_ = EP_generator(P, DS, DS)
#         EP = numpy.zeros(len(P) - DS + 1)
#         EP[0] = next(EP_)
#         for i in range(1, len(P) - DS + 1):
#             EP[i] = EP_.send([i, DS])
#         return EP
#         EP_.close()
#
# starttime2 = time.time()
# EP2 = fun_EP2(P, DS)
# endtime2 = time.time()
# dtime2 = endtime2 - starttime2
# print('dtim2:{}'.format(dtime2))
#
#
# def fun_EP(P, DS=365):
#     '''计算有效降水'''
#     if type(DS) is numpy.ndarray:  # 当传入DS为数组时
#         EP = numpy.zeros(len(P) - DS[0] + 1)
#         for i in range(len(P) - DS[0] + 1):
#             s_ = 0
#             for j in range(1, DS[i]):
#                 s_ += P[i + DS[0] - 1 - j:i + DS[0]].mean()
#             EP[i] = s_ + P[i + DS[0] - 1]
#         return EP
#     else:
#         EP = numpy.zeros(len(P) - DS + 1)
#         for i in range(len(P) - DS + 1):
#             s_ = 0
#             for j in range(1, DS):
#                 s_ += P[i + DS - 1 - j:i + DS].mean()
#             EP[i] = s_ + P[i + DS - 1]
#         return EP
# starttime1 = time.time()
# EP1 = fun_EP(P, DS)
# endtime1 = time.time()
# dtime1 = endtime1 - starttime1
# print('dtim1:{}'.format(dtime1))

# result = (EP2 - EP1)
# print(numpy.argwhere(result>0.001))

# def fun_EP(P, DS=365):
#     '''计算有效降水'''
#     if type(DS) is numpy.ndarray:  # 当传入DS为数组时
#         EP = numpy.zeros(len(P) - DS[0] + 1)
#         for i in range(len(P) - DS[0] + 1):
#             s_ = 0
#             for j in range(DS[i]):
#                 s_ += P[i + DS[0] - 1 - j:i + DS[0]].mean()
#             EP[i] = s_
#         return EP
#     else:
#         # EP = numpy.array([P[i + DS - 1 - j:i + DS].mean() for i in range(len(P) - DS + 1) for j in range(DS)]).reshape(
#         #     len(P) - DS + 1, DS).sum(axis=1)
#         # return EP
#         EP = numpy.zeros(len(P) - DS + 1)
#         for i in range(len(P) - DS + 1):
#             s_ = 0
#             for j in range(DS):
#                 s_ += P[i + DS - 1 - j:i + DS].mean()
#             EP[i] = s_
#         return EP
# 已验证
# starttime1 = time.time()
# EP1 = fun_EP(P, DS)
# endtime1 = time.time()
# dtime1 = endtime1 - starttime1
# print('dtim1:{}'.format(dtime1))
# EP3=[P[i + DS[0] - 1 - j:i + DS[0]].mean() for i in range(len(P) - DS[0] + 1) for j in range(DS[i])]

# def fun_EP3(P, DS=365):
#     if type(DS) is numpy.ndarray:  # 当传入DS为数组时
#         EP_ = [P[i + DS[0] - 1 - j:i + DS[0]].mean() for i in range(len(P) - DS[0] + 1) for j in range(DS[i])]
#         EP = numpy.array([sum(EP_[sum(DS[:i-1]):sum(DS[:i])]) for i in range(1, len(P) - DS[0] + 1)])
#         numpy.insert(EP, 0, sum(EP_[:DS[0]]))
#         return EP
#     else:
#         EP = numpy.array([P[i + DS - 1 - j:i + DS].mean() for i in range(len(P) - DS + 1) for j in range(DS)]).reshape(
#             len(P) - DS + 1, DS).sum(axis=1)
#         return EP
# starttime3 = time.time()
# EP3=fun_EP3(P, DS)
# endtime3 = time.time()
# dtime3 = endtime3 - starttime3
# print('dtim3:{}'.format(dtime3))
# [::DS].sum()

# def fun_EP2(P, DS=365):
#     '''计算有效降水'''
#     if type(DS) is numpy.ndarray:  # 当传入DS为数组时
#         EP = numpy.zeros(len(P) - DS[0] + 1)
#         for i in range(len(P) - DS[0] + 1):
#             # s_ = 0
#             # for j in range(DS[i]):
#             #     s_ += P[i + DS[0] - 1 - j:i + DS[0]].mean()
#             # EP[i] = s_
#             EP[i] = sum([P[i + DS[0] - 1 - j:i + DS[0]].mean() for j in range(DS[i])])
#         return EP
#     else:
#         EP = numpy.zeros(len(P) - DS + 1)
#         for i in range(len(P) - DS + 1):
#             # s_ = 0
#             # for j in range(DS):
#             #     s_ += P[i + DS - 1 - j:i + DS].mean()
#             # EP[i] = s_
#             EP[i] = sum([P[i + DS - 1 - j:i + DS].mean() for j in range(DS)])
#         return EP
# EP2 = fun_EP2(P, DS)

