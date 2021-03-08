# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Spatial analysis: fixed time (one time), as a map
import numpy as np
import pandas as pd
import FDIP
import os
from matplotlib import pyplot as plt
import cartopy_plot

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
static_params = pd.read_excel(os.path.join(home, "5.Analysis_spatial/static_params.xlsx"), index_col=0)

# set for spatial plot
lon = coord.loc[:, "lon"].values
lat = coord.loc[:, "lat"].values
det = 0.25
lat_min = min(lat)
lon_min = min(lon)
lat_max = max(lat)
lon_max = max(lon)
extend = [lon_min, lon_max, lat_min, lat_max]
shape_file = [f"{root}:/GIS/Flash_drought/f'r_project.shp"]

# spatial plot of time average sm
sm_rz_time_avg = sm_rz.mean(axis=0)
cartopy_plot.general_cartopy_plot(extend, det, sm_rz_time_avg, lat, lon, shape_file=shape_file, expand=5, grid=True,
                                  save=False, title="Spatial Distribution of Time-avg Soil Moisture",
                                  cb_label="Soil Moisture", cmap_name="RdBu")

# spatial plot of Drought params
cartopy_plot.general_cartopy_plot(extend, det, static_params["Drought_FOC"].values, lat, lon, shape_file=shape_file,
                                  expand=5, grid=True, save=False, title="Spatial Distribution of Drought FOC",
                                  cb_label="frequency", cmap_name="YlOrBr")
cartopy_plot.general_cartopy_plot(extend, det, static_params["Drought_number"].values, lat, lon, shape_file=shape_file,
                                  expand=5, grid=True, save=False, title="Spatial Distribution of Drought number",
                                  cb_label="number", cmap_name="YlOrRd")
cartopy_plot.general_cartopy_plot(extend, det, static_params["DD_mean"].values, lat, lon, shape_file=shape_file,
                                  expand=5, grid=True, save=False, title="Spatial Distribution of mean Drought Duration",
                                  cb_label="pentad", cmap_name="Oranges")
cartopy_plot.general_cartopy_plot(extend, det, static_params["DS_mean"].values, lat, lon, shape_file=shape_file,
                                  expand=5, grid=True, save=False, title="Spatial Distribution of mean Drought Severity",
                                  cb_label="pentad*mm", cmap_name="Reds")
cartopy_plot.general_cartopy_plot(extend, det, static_params["SM_min_mean"].values, lat, lon, shape_file=shape_file,
                                  expand=5, grid=True, save=False, title="Spatial Distribution of mean SM_min during drought",
                                  cb_label="mm", cmap_name="BrBG")





