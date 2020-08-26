# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com

import EDI_sample
import EDI_refactoring
import numpy
from climate_indices import indices
from matplotlib import pyplot as plt
import pandas
import math, pandas
import datetime
import time


P = numpy.loadtxt('F:/小论文2\代码/results/daily_gldas/Rainf_f_tavg_array_daily.txt')
# 单位转换  "kg m-2 s-1"，换算系数3600*24
P = P*3600*24

# 获取EDI时间起始/结束/长度(用于创建下面的矩阵存)(因为EDI计算过程中有些时间是消掉了)，用一个降水数据计算EDI_refactoring，得到数据组来看
# DATA1 = EDI_refactoring.EDI(P[:, 0], start=[1948, 1, 1], end=[2014, 12, 30], h_sep=0, DS=30, freq='D')
# DATA2 = EDI_sample.EDI_only(P[:, 0], h_sep=0, DS=30)
# EDI365:19481230-20141213，长度24090
# EDI30:19480130-20140112，长度24090

# 计算EDI, 这一步耗时很久, DS=30(1min 6次)
EDI = numpy.full((1057, 24090), -9999).astype('float')
for i in range(1057):
    EDI[i, :] = EDI_sample.EDI_only(P[:, i], h_sep=0, DS=30)
    print(i)

EDI = EDI.T
numpy.savetxt('F:/小论文2/代码/results/daily_gldas/EDI_daily_30.txt', EDI)


# 读取EDI
EDI = numpy.loadtxt('F:/小论文2/代码/results/daily_gldas/EDI_daily_30.txt')
EDI_pd = pandas.DataFrame(EDI, index=pandas.date_range(start='19480130', end='20140112', freq='D'))

# EDI存到excel
# EDI_pd.loc["1948-12-30":"1949-01-08", :].to_excel('F:/小论文2/代码/results/daily_gldas/EDI_daily_ceshi.xlsx')


# 构建EDI-nc文件


# 计算SPI
# SPI = numpy.full((1057, 24471), -9999).astype('float')
# Periodicity = indices.compute.Periodicity.daily
# Distribution = indices.Distribution.gamma
#
# for i in range(1057):
#     SPI[i, :] = indices.spi(P[:, i], scale=365, periodicity=Periodicity, data_start_year=1948, calibration_year_initial=1948,
#                        calibration_year_final=2014, distribution=Distribution)
#
# numpy.savetxt('F:/小论文2/代码/results/daily_gldas/SPI_daily_365.txt', SPI)

# 读取SPI
# SPI = numpy.loadtxt('F:/小论文2/代码/results/daily_gldas/SPI_daily_365.txt')
# SPI_pd = pandas.DataFrame(SPI.T, index=pandas.date_range(start='19480101', end='20141230', freq='D'))

# SPI存到excel
# SPI_pd.loc[EDI_pd.iloc[:10, :].index, :].to_excel('F:/小论文2/代码/results/daily_gldas/SPI_daily_ceshi_365.xlsx')

# 计算区域统计值
# P_mean = P.T.mean(axis=0)
# EDI_mean = EDI.mean(axis=0)
# SPI_mean = SPI.mean(axis=0)


# 读取时间，构建dataframe来绘图,查看源数据


# 绘图
# SPI_mean_df.plot()
# P_mean_df.plot()

# 读取经纬度
# coord = pandas.read_csv('results/daily_gldas/coord.txt')