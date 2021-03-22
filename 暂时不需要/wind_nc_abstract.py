# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd

home = 'F:/work/jianglong'
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
times = f1.variables["time"][:]
u = np.zeros((len(times), len(coord)), dtype=float)
v = np.zeros((len(times), len(coord)), dtype=float)

for i in range(len(times)):
    for j in range(len(coord)):
        u[i, j] = f1.variables[variable_name_u][i, lat_index[j], lon_index[j]]
        v[i, j] = f1.variables[variable_name_v][i, lat_index[j], lon_index[j]]

# cal winAbs and winDir
''' 
u: 正西风，朝右 
v: 正南风，朝上
'''
windSpeed = (u ** 2 + v ** 2) ** (0.5)
windDirection = np.arctan(v / u)

ret_u = pd.DataFrame(u, index=times, columns=list(range(len(coord))))
ret_v = pd.DataFrame(v, index=times, columns=list(range(len(coord))))
ret_windSpeed = pd.DataFrame(windSpeed, index=times, columns=list(range(len(coord))))
ret_windDirection = pd.DataFrame(windDirection, index=times, columns=list(range(len(coord))))


# f1.close()
# # define variable
# variable = np.zeros((1, coord_number + 1))
#
#
# # read variable based on the lat_index/lon_index
# for i in range(len(result)):
#     f = Dataset(result[i], 'r')
#     Dataset.set_auto_mask(f, False)
#     time_number = f.variables["time"].shape[0]
#     variable_ = np.zeros((time_number, coord_number + 1))
#     variable_[:, 0] = f.variables["time"][:]
#     for j in range(len(coord)):
#         for k in range(len(f.variables["time"])):
#             variable_[k, j + 1] = f.variables[variable_name][k, lat_index[j], lon_index[j]]
#     print(f"complete read file:{i}")
#     variable = np.vstack((variable, variable_))
#     f.close()
#
# variable = variable[1:, :]
#
# # sort by time
# variable = variable[variable[:, 0].argsort()]
# save
# np.savetxt(f'{variable_name}.txt', variable, delimiter=' ')
# coord.to_csv("coord.csv")