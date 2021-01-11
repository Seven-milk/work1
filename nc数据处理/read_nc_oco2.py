# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
import numpy as np

home = "H:/data_zxd/oco2"
data_path = [os.path.join(home, data_) for data_ in os.listdir(home) if data_[-4:] == ".nc4"]

# data0 = nc.Dataset(data_path[0])
# lat0 = data0.variables['latitude'][:]
# lon0 = data0.variables['longitude'][:]
# xco20 = data0.variables['xco2'][:]
# data0.close()

data_result = []

for i in range(len(data_path)):
    data_nc = nc.Dataset(data_path[i])
    lat = data_nc.variables['latitude'][:]
    lon = data_nc.variables['longitude'][:]
    xco2 = data_nc.variables['xco2'][:]
    result = np.vstack((lat, lon, xco2))
    data_result.append(result)
    data_nc.close()


# 画散点图
plt.figure()
# plt.boxplot(xco2)
plt.scatter(data_result[0][1, :], data_result[0][0, :], s=3, c=data_result[0][2, :], cmap='YlOrRd', vmax=450)
plt.colorbar()
plt.show()
