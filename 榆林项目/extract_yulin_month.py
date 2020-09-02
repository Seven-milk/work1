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


# 单位处理
# 单位转换  "kg m-2 s-1"="mm/s"=3600*24*30"mm/month"，换算系数3600*24*30（按每个月30天算）
# 单位转换 "kg m-2"="mm"=30*24*3600"mm/month",换算系数30*24*3600（因为是月平均值，换算成月累计值）
Rainf_f_tavg_array = Rainf_f_tavg_array*3600*24*30
Qs_acc_array = Qs_acc_array*3600*24*30


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


# 搭建pd来输出excel
time = pd.date_range(start="19480101", end="20141231", freq='M')
Rainf_f_tavg_pd = pd.DataFrame(Rainf_f_tavg_array, index=time)
Qs_acc_array_pd = pd.DataFrame(Qs_acc_array, index=time)
Rainf_f_tavg_pd = Rainf_f_tavg_pd.loc["1960-01":"2010-12", :]
Qs_acc_array_pd = Qs_acc_array_pd.loc["1960-01":"2010-12", :]


# 输出excel
Rainf_f_tavg_pd.to_excel("Rainf_f_tavg_pd.xlsx")
Qs_acc_array_pd.to_excel("Qs_acc_array_pd.xlsx")