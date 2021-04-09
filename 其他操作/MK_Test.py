# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import math
import numpy as np


def MK_test(x):
    s = 0
    length = len(x)
    for m in range(0, length - 1):
        print(m)
        print('/')
        for n in range(m + 1, length):
            print(n)
            print('*')
            if x[n] > x[m]:
                s = s + 1
            elif x[n] == x[m]:
                s = s + 0
            else:
                s = s - 1
    # 计算vars
    vars = length * (length - 1) * (2 * length + 5) / 18
    # 计算zc
    if s > 0:
        zc = (s - 1) / math.sqrt(vars)
    elif s == 0:
        zc = 0
    else:
        zc = (s + 1) / math.sqrt(vars)

    # 计算za
    zc1 = abs(zc)

    # 计算倾斜度
    ndash = length * (length - 1) // 2
    slope1 = np.zeros(ndash)
    m = 0
    for k in range(0, length - 1):
        for j in range(k + 1, length):
            slope1[m] = (x[j] - x[k]) / (j - k)
            m = m + 1

    slope = np.median(slope1)
    return (slope, zc1)


# x = np.loadtxt("x.txt")
# (slope, zc1) = MK_test(x)