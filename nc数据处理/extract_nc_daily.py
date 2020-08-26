# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 处理nc文件，将GLDASmonthly 0.25 at GES DISC 1948-2014读取lp范围的 Rainf_f_tavg 数据到数组中
# 结果为Rainf_f_tavg_array_month.pkl
# 有空值, 已处理

import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd


coord = pd.read_csv("F:/data/arcgis数据/黄土高原空间范围/coordi_xy.txt", sep=",")  # 读取coord黄河流域的经纬坐标（基于渔网提取）
coord = coord.round(3)  # 处理单位以便与nc中lat lon一致
result = os.listdir('G:/GLADS/data')
Rainf_f_tavg_array = np.zeros((len(result), len(coord)))


# 获取lp经纬的索引
f1 = Dataset('G:/GLADS/data/{}'.format(result[1]), 'r')
Dataset.set_auto_mask(f1, False)
lat_index_lp = []
lon_index_lp = []
lat = f1.variables["lat"][:]
lon = f1.variables["lon"][:]
for j in range(len(coord)):
    lat_index_lp.append(np.where(lat == coord["lat"][j])[0][0])
    lon_index_lp.append(np.where(lon == coord["lon"][j])[0][0])
f1.close()


# 循环读取对应lp范围的rain变量(基于索引)
for i in range(len(result)):
    f = Dataset('G:/GLADS/data/{}'.format(result[i]), 'r')
    Dataset.set_auto_mask(f, False)
    for j in range(len(coord)):
        Rainf_f_tavg_array[i, j] = f.variables['Rainf_f_tavg'][0, lat_index_lp[j], lon_index_lp[j]]
        # 读取黄河流域部分,只取相交部分1057个，所以不能用f.variables['Rainf_f_tavg'][0, lat_index_lp, lon_index_lp]（1057*1057）
    f.close()


# 存储数据
np.savetxt('Rainf_f_tavg_array_daily.txt', Rainf_f_tavg_array, delimiter=' ')
np.savetxt('lat_index_lp.txt', lat_index_lp, delimiter=' ')
np.savetxt('lon_index_lp.txt', lon_index_lp, delimiter=' ')
coord.to_csv("coord.txt")