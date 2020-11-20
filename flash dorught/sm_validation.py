# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# validation the sm data
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg'

import numpy as np
import pandas as pd
import FDIP
import os
import re
from matplotlib import pyplot as plt

# soil moisture data validation
# home = "H:/research/flash_drough/"
# data_path = os.path.join(home, "GLDAS_Catchment")
# coord_path = "H:/GIS/Flash_drought/coord.txt"
# coord = pd.read_csv(coord_path, sep=",")
# date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
# sm_rz = np.loadtxt(os.path.join(data_path, "SoilMoist_RZ_tavg.txt"), dtype="float", delimiter=" ")

# sm from NETWORK
sm_network_path = "H:/data_zxd/LP/SM_ISMN"
network_china = os.path.join(sm_network_path, "CHINA")

# china read txt to excel
stations = os.listdir(network_china)
years = list(range(1981, 2000))
months = list(range(1, 13))
days = [8, 18, 28]
pd_index = [f"{year}/{month}/{day}" for year in years for month in months for day in days]
pd_index = pd.to_datetime(pd_index)
# for station in stations:
#     result = pd.DataFrame(np.full((len(pd_index), 11), fill_value=np.NAN), index=pd_index)
#     stms = [os.path.join(network_china, station, d) for d in os.listdir(os.path.join(network_china, station)) if
#             d[-4:] == ".stm"]
#     for i in range(len(stms)):
#         with open(stms[i]) as f:
#             str_ = f.read()
#         str_ = str_.splitlines()
#         index_ = pd.to_datetime([i[:10] for i in str_[2:]])
#         data_ = pd.Series([float(i[19:25]) for i in str_[2:]], index=index_)
#         for j in range(len(data_)):
#             result.loc[data_.index[j], i] = data_.loc[data_.index[j]]
#     result.to_excel(f"{station}.xlsx")

names = locals()
stations = [os.path.join(network_china, d) for d in stations if d[-5:] == ".xlsx"]
for station in stations:
    names[re.search(r"\\[A-Z]*\.xlsx", station)[0][1:-5]] = pd.read_excel(station, index_col=0)
