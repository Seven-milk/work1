# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd
import time
import re

home = 'H:/work/daimeng'
path = os.path.join(home, 'soil moisture layer 1')
# path = os.path.join(home, 'soil moisture layer 2')
coord_path = os.path.join(home, 'coord.txt')
precision = 3
variable_name = 'sm1'
# variable_name = 'sm2'
coord = pd.read_csv(coord_path, sep=",")  # read coord(extract by fishnet)
coord = coord.round(precision)  # coord precision correlating with .nc file lat/lon
coord_number = len(coord)
result = [path + "/" + d for d in os.listdir(path) if d[-3:] == ".nc"]
print(f"file number:{len(result)}")

# calculate the index of lat/lon in coord from source nc file
f1 = Dataset(result[0], 'r')
Dataset.set_auto_mask(f1, False)
lat_index = []
lon_index = []
lat = f1.variables["lat"][:]
lon = f1.variables["lon"][:]
for j in range(len(coord)):
    lat_index.append(np.where(lat == coord["lat"][j])[0][0])
    lon_index.append(np.where(lon == coord["lon"][j])[0][0])
f1.close()

# read variable based on the lat_index/lon_index
for i in range(len(result)):
    f = Dataset(result[i], 'r')
    Dataset.set_auto_mask(f, False)
    time_number = f.variables["time"].shape[0]
    variable_ = np.zeros((time_number, coord_number + 1))
    variable_[0, :] = f.variables["time"][:]
    for j in range(len(coord)):
        for k in len(f.variables["time"]):
            variable_[i, j + 1] = f.variables[variable_name][f.variables["time"][k], lat_index[j], lon_index[j]]
    print(f"complete read file:{i}")
    variable = np.vstack((variable, variable_))
    f.close()

# sort by time
variable = variable[variable[:, 0].argsort()]
# save
np.savetxt(f'{variable_name}.txt', variable, delimiter=' ')
np.savetxt('lat_index.txt', lat_index, delimiter=' ')
np.savetxt('lon_index.txt', lon_index, delimiter=' ')
coord.to_csv("coord.txt")