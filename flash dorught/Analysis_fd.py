# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# analysis flash drought
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg'

import numpy as np
import pandas as pd
import FDIP
import os
import re
from matplotlib import pyplot as plt

home = "H:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment")
coord_path = "H:/GIS/Flash_drought/coord.txt"
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101','20141230',freq='d').strftime("%Y%m%d").to_numpy(dtype="int")

# soil moisture data validation
sm_rz = np.loadtxt(os.path.join(data_path, "SoilMoist_RZ_tavg.txt"), dtype="float", delimiter=" ")

# pretreatment: pentad sm_rz
num_pentad = len(date)//5
num_out = len(date) - num_pentad * 5
sm_rz = sm_rz[:-1]
date = date[:-1]
sm_rz_pentad = np.full((num_pentad, len(coord)), fill_value=-9999, dtype="float")
for i in range(num_pentad):
    sm_rz_pentad[i, :] = sm_rz[i*5:(i+1)*5, :].mean(axis=0)

# spaital average sm
sm_rz_pend_avg = sm_rz_pentad.mean(axis=1)
FD_avg = FDIP.FD(sm_rz_pend_avg, timestep=73, Date=date, threshold1=0.4, threshold2=0.2, RI_threshold=0.1,
                 RI_mean_threshold=0.065)

