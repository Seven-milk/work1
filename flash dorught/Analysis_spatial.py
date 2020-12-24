# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# plot the spatial distribution of FD
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
sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")

# time avg data
sm_rz_time_avg = sm_rz.mean(axis=0)

# plot the spatial distribution of time avg data
lon = coord.loc[:, "lon"].values
lat = coord.loc[:, "lat"].values
det = 0.25

lat_min = min(lat)
lon_min = min(lon)
lat_max = max(lat)
lon_max = max(lon)
extend = [lon_min, lon_max, lat_min, lat_max]
shape_file=["H:/GIS/Flash_drought/f'r_project.shp"]
cartopy_plot.general_cartopy_plot(extend, det, sm_rz_time_avg, lat, lon, shape_file=shape_file, expand=5, grid=True, save=False, title="Map", cb_label="Soil Moisture")
