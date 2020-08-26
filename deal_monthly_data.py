# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

import netCDF4 as nc
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy, pandas
from matplotlib import pyplot as plt
from netCDF4 import Dataset
from climate_indices import indices
import os
import dill
import pandas as pd

P = numpy.loadtxt('results/month_gldas/Rainf_f_tavg_array_month.txt')


# 单位转换  "kg m-2 s-1"，换算系数3600*24
P = P*3600*24


# 计算SPI-1
# SPI = numpy.full((1057, 804), -9999).astype('float')
# Periodicity = indices.compute.Periodicity.monthly
# Distribution = indices.Distribution.gamma
#
# for i in range(1057):
#     SPI[i, :] = indices.spi(P[:, i], scale=1, periodicity=Periodicity, data_start_year=1948, calibration_year_initial=1948,
#                        calibration_year_final=2014, distribution=Distribution)
#
# numpy.savetxt('F:/小论文2/代码/results/month_gldas/SPI_monthly.txt', SPI)

# 读取SPI
SPI = numpy.loadtxt('F:/小论文2/代码/results/month_gldas/SPI_monthly.txt')


# 读取时间，构建dataframe来绘图,查看源数据 0101
time = pandas.date_range(start='19480101', end='20141231', freq='M')

# 统计空间平均
SPI_mean = SPI.mean(axis=0)

# 绘图
plt.plot(time, SPI_mean)