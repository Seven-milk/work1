# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 牌子编码计算
import numpy as np

start = int(input("输入开始数字:"))
str1 = str(input("输入要加的开头字符:"))
end = start + 720
x1 = list(range(start, end))
# x2 = list(range(10, 100))
# x3 = list(range(100,360))
y = []
for i in range(len(x1)):
    y.append(str1 + str(x1[i]))
# for i in range(len(x2)):
#     y.append('0' + str(x2[i]))
# for i in range(len(x3)):
#     y.append(str(x3[i]))

z = np.array(y)
np.savetxt('数据源{}.txt'.format(start), z, fmt='%s', delimiter=',')