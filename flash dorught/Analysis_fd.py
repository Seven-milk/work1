# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# analysis flash drought
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg'

import numpy as np
import pandas as pd
import FDIP
import os
from matplotlib import pyplot as plt

home = "H:/research/flash_drough/"
data_path = os.path.join(home, "data")
coord_path = "H:/GIS/Flash_drought/coord.txt"
coord = pd.read_csv(coord_path, sep=",")

# soil moisture data validation
sm_rz = np.loadtxt(os.path.join(data_path, "SoilMoist_RZ_tavg.txt"), dtype="float", delimiter=" ")

# pretreatment
# smooth

# distrbution
# hist
for i in range(len()):

# Identify FD event based on average sm
sm_rz_avg = sm_rz.mean(axis=1)
