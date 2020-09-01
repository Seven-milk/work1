# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 处理nc文件，将GLDASmonthly 0.25 at GES DISC 1948-2014读取榆林范围的 Rainf_f_tavg-precipitation_flux 和
# Qs_acc-surface_runoff_amount 数据到数组中
# 结果为Rainf_f_tavg_array_month.pkl/txt

import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd

home = 'D:/GLADS/GLDASmonthly 0.25 at GES DISC 1948-2014/downthemall1948-2014'
coord = pd.read_csv("H:/data/榆林市/yulin_coord.txt", sep=",")  # 读取coord榆林的经纬坐标（基于渔网提取）
coord = coord.round(3)  # 处理单位以便与nc中lat lon一致
result = os.listdir(home)
Rainf_f_tavg_array = np.zeros((len(result), len(coord)))
Qs_acc_array = np.zeros((len(result), len(coord)))

# 获取榆林经纬的索引
f1 = Dataset(home + "/" + result[0], 'r')
Dataset.set_auto_mask(f1, False)
lat_index_lp = []
lon_index_lp = []
lat = f1.variables["lat"][:]
lon = f1.variables["lon"][:]
for j in range(len(coord)):
    lat_index_lp.append(np.where(lat == coord["lat"][j])[0][0])
    lon_index_lp.append(np.where(lon == coord["lon"][j])[0][0])
f1.close()

# 循环读取对应lp范围的rain变量和runoff变量(基于索引)
for i in range(len(result)):
    f = Dataset(home + "/" + result[i], 'r')
    Dataset.set_auto_mask(f, False)
    for j in range(len(coord)):
        Rainf_f_tavg_array[i, j] = f.variables['Rainf_f_tavg'][0, lat_index_lp[j], lon_index_lp[j]]
        Qs_acc_array[i, j] = f.variables['Qs_acc'][0, lat_index_lp[j], lon_index_lp[j]]
        # 读取榆林部分
    print(f"第{i}个栅格读取完毕------")
    f.close()


# 存储数据
np.savetxt('Rainf_f_tavg_array_month.txt', Rainf_f_tavg_array, delimiter=' ')
np.savetxt('Qs_acc.txt', Qs_acc_array, delimiter=' ')
np.savetxt('lat_index_lp.txt', lat_index_lp, delimiter=' ')  # 存储经纬索引
np.savetxt('lon_index_lp.txt', lon_index_lp, delimiter=' ')
# np.savetxt('coord.txt', coord, delimiter=' ')
coord.to_csv("coord.txt")


# 概览
# rootgrp = Dataset(home + "/GLDAS_NOAH025_M.A194801.020.nc4", "r")
# print(rootgrp.variables.keys())
# print('****************************')
# print(rootgrp)
# print('****************************')
# print(rootgrp.variables['lat'][:])
# print('****************************')
# print(rootgrp.variables['lon'][:])
# rootgrp.close()
