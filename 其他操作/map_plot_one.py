# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd

def map_plot(filename, variable_name):
    map = Basemap()  # 创建basemap对象
    data = Dataset(filename)
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lons, lats = map.makegrid(len(lat), len(lon))
    # lats = lats[::-1]
    data.set_auto_mask(True)
    data1 = data.variables[variable_name][:].reshape(len(lon), len(lat))


    map.drawmapboundary(fill_color='aqua')  # 绘制底图
    # map.fillcontinents(color='#cc9955', lake_color='aqua')  # 绘制陆地
    map.drawcoastlines(linewidth=0.2, color='0.15')  # 绘制海岸线
    map.drawcountries(linewidth=0.2, color='0.15')  # 绘制城市
    map.contourf(lons, lats, data1)  # 绘制数据
    map.colorbar()
    plt.show()

if __name__ == '__main__':
    filename = 'F:/GLDAS_NOAH025_M.A194801.020.nc4'
    variable_name = 'Rainf_f_tavg'
    map_plot(filename, variable_name)