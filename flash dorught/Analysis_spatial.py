# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Spatial analysis: fixed time (one time), as a map
import numpy as np
import pandas as pd
import FDIP
import os
import re
from matplotlib import pyplot as plt
from cartopy import crs
from cartopy import feature
import cartopy_plot

# general set
home = "H:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment/SoilMoist_RZ_tavg.txt")
coord_path = os.path.join(home, "coord.txt")
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
date_pentad = np.loadtxt(os.path.join(home, "date_pentad.txt"), dtype="int")
sm_rz_pentad = np.loadtxt(os.path.join(home, "sm_rz_pentad.txt"))
sm_percentile_rz_pentad = np.loadtxt(os.path.join(home, "sm_percentile_rz_pentad.txt"), dtype="float", delimiter=" ")

# set for plot
lon = coord.loc[:, "lon"].values
lat = coord.loc[:, "lat"].values
det = 0.25
lat_min = min(lat)
lon_min = min(lon)
lat_max = max(lat)
lon_max = max(lon)
extend = [lon_min, lon_max, lat_min, lat_max]
shape_file = ["H:/GIS/Flash_drought/f'r_project.shp"]

# time average sm: Analysis
sm_rz_time_avg = sm_rz.mean(axis=0)
cartopy_plot.general_cartopy_plot(extend, det, sm_rz_time_avg, lat, lon, shape_file=shape_file, expand=5, grid=True,
                                      save=False, title="Spatial Distribution of Time-avg Soil Moisture",
                                      cb_label="Soil Moisture")

# calculate the statistical parameters of each point
# for i in range(sm_rz.shape[1]):
#     FD_ = FDIP.FD(sm_rz_pentad, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5, pc=0.28,
#                   excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
#                   fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
