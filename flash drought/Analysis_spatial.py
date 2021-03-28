# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Spatial analysis: fixed time (one time), as a map
import numpy as np
import pandas as pd
import FDIP
import os
from matplotlib import pyplot as plt
from importlib import reload
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
res_grid = 1
res_label = 2
lat_min = min(lat)
lon_min = min(lon)
lat_max = max(lat)
lon_max = max(lon)
extent = [lon_min, lon_max, lat_min, lat_max]
boundry_shp = [f"{root}:/GIS/Flash_drought/f'r_project.shp"]
Helong_shp = [f"{root}:/GIS/Flash_drought/he_long_projection.shp"]
boundry_shpMap = cartopy_plot.ShpMap(boundry_shp)
Helong_shpMap = cartopy_plot.ShpMap(Helong_shp, linestyle="--", linewidth=0.5)

# spatial plot of time average sm
sm_rz_time_avg = sm_rz.mean(axis=0)


def plot_sm():
    fig_sm = cartopy_plot.Figure()
    map_sm = cartopy_plot.Map(fig_sm.ax, fig_sm, extent=extent, grid=True, res_grid=res_grid, res_label=res_label,
                              title="Spatial Distribution of Time-avg Soil Moisture")
    raster_sm = cartopy_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=sm_rz_time_avg,
                                       expand=5,
                                       cmap_name='RdBu', cb_label="Soil Moisture / mm")
    map_sm.addmap(raster_sm)
    map_sm.addmap(Helong_shpMap)
    map_sm.addmap(boundry_shpMap)


plot_sm()


# spatial plot of Drought params
def plot_drought_params():
    fig_drought_params = cartopy_plot.Figure(5)
    title = ["Drought FOC", "Drought number", "Mean Drought Duration", "Mean Drought Severity", "Mean SM_min"]
    cb_label = ["frequency", "number", "pentad", "pentad*\npercentile", "percentile"]
    data = [static_params["Drought_FOC"].values, static_params["Drought_number"].values,
            static_params["DD_mean"].values,
            static_params["DS_mean"].values, static_params["SM_min_mean"].values]
    for i in range(len(title)):
        map_ = cartopy_plot.Map(fig_drought_params.ax[i], fig_drought_params, extent=extent, grid=True,
                                res_grid=res_grid,
                                res_label=res_label, title=title[i])
        map_.addmap(boundry_shpMap)
        map_.addmap(Helong_shpMap)
        raster_ = cartopy_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=data[i], expand=5,
                                         cmap_name='BrBG', cb_label=cb_label[i])
        map_.addmap(raster_)


plot_drought_params()


# spatial plot of FD params
def plot_FD_params():
    fig_FD_params = cartopy_plot.Figure(6)
    title = ["FD FOC", "FD number", "Mean FD Duration", "Mean FD Severity", "Mean RI_mean", "Mean RI_max"]
    cb_label = ["frequency", "number", "pentad", "pentad*\npercentile", "percentile/\npentad", "percentile/\npentad"]
    data = [static_params["FD_FOC"].values, static_params["FD_number"].values, static_params["FDD_mean"].values,
            static_params["FDS_mean"].values, static_params["RI_mean_mean"].values, static_params["RI_max_mean"].values]
    for i in range(len(title)):
        map_ = cartopy_plot.Map(fig_FD_params.ax[i], fig_FD_params, extent=extent, grid=True, res_grid=res_grid,
                                res_label=res_label, title=title[i])
        map_.addmap(boundry_shpMap)
        map_.addmap(Helong_shpMap)
        raster_ = cartopy_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=data[i], expand=5,
                                         cmap_name='BrBG', cb_label=cb_label[i])
        map_.addmap(raster_)


plot_FD_params()
