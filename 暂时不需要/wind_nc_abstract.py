# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd

home = 'H:/work/jianglong'
path = os.path.join(home, 'jianglong.nc')
precision = 3
variable_name_u = 'u10'
variable_name_v = 'v10'

# get lat/lon - point number file
f1 = Dataset(path, 'r')
Dataset.set_auto_mask(f1, False)
lats = f1.variables["latitude"][:]
lons = f1.variables["longitude"][:]
coord = pd.DataFrame(columns=('lat', 'lon'))
for lat in lats:
    for lon in lons:
        coord = coord.append(pd.DataFrame({'lat':[lat],'lon':[lon]}))
coord.index = list(range(len(coord)))

# calculate the index of lat/lon in coord from source nc file
lat_index = []
lon_index = []
for j in range(len(coord)):
    lat_index.append(np.where(lats == coord["lat"][j])[0][0])
    lon_index.append(np.where(lons == coord["lon"][j])[0][0])

# get u and v
Dataset.set_auto_mask(f1, True)
times = f1.variables["time"][:]
u = np.zeros((len(times), len(coord)), dtype=float)
v = np.zeros((len(times), len(coord)), dtype=float)

for i in range(len(times)):
    for j in range(len(coord)):
        u[i, j] = f1.variables[variable_name_u][i, lat_index[j], lon_index[j]]
        v[i, j] = f1.variables[variable_name_v][i, lat_index[j], lon_index[j]]
        print(f"complete {j}")

# cal winAbs and winDir
''' 
u: 正西风，朝右 
v: 正南风，朝上
'''
windSpeed = (u ** 2 + v ** 2) ** (0.5)
windDirection = np.arctan(v / u)  # -pi/2 ~ pi/2, 可以转换为pi单位的角度值

ret_u = pd.DataFrame(u, index=times, columns=list(range(len(coord))))
ret_v = pd.DataFrame(v, index=times, columns=list(range(len(coord))))
ret_windSpeed = pd.DataFrame(windSpeed, index=times, columns=list(range(len(coord))))
ret_windDirection = pd.DataFrame(windDirection, index=times, columns=list(range(len(coord))))


# find nan index
index_nan = []
for i in range(len(coord)):
    if np.isnan(u[0, i]):
        index_nan.append(i)

# delete index_nan in coord/ ret_u/ ret_v/ ret_windSpeed/ ret_windDirection = 713 - 421 = 292
for index in index_nan:
    ret_u.drop(columns=index, inplace=True)
    ret_v.drop(columns=index, inplace=True)
    ret_windSpeed.drop(columns=index, inplace=True)
    ret_windDirection.drop(columns=index, inplace=True)
    coord.drop(index=index, inplace=True)

# time
time_index = pd.date_range('19810101 06:00:00', '19821231 18:00:00', freq='6H')
ret_u.index = time_index
ret_v.index = time_index
ret_windSpeed.index = time_index
ret_windDirection.index = time_index

# save
ret_u.to_excel('u10.xlsx')
ret_v.to_excel('v10.xlsx')
ret_windSpeed.to_excel('windSpeed.xlsx')
ret_windDirection.to_excel('windDirection.xlsx')
coord.to_excel('coord.xlsx')

f1.close()