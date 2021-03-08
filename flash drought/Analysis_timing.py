# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Timing analysis: fixed spatial (one region), as a time series
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg'
import numpy as np
import pandas as pd
import FDIP
import os
from matplotlib import pyplot as plt

# general set
root = "H"
home = f"{root}:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment/SoilMoist_RZ_tavg.txt")
coord_path = os.path.join(home, "coord.txt")
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
date_pentad = np.loadtxt(os.path.join(home, "date_pentad.txt"), dtype="int")
sm_rz_pentad = np.loadtxt(os.path.join(home, "sm_rz_pentad.txt"))
sm_percentile_rz_pentad = np.loadtxt(os.path.join(home, "sm_percentile_rz_pentad.txt"), dtype="float", delimiter=" ")
Num_point = 1166  # grid number

# time series of spaital average sm
sm_rz_pentad_spatial_avg = sm_rz_pentad.mean(axis=1)
FD_avg = FDIP.FD(sm_rz_pentad_spatial_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5, pc=0.28,
                 excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                 fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
SM_percentile, RI, out_put, dp = FD_avg.general_out()
# out_put.to_excel("out_put.xlsx")

# hot season