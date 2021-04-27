# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Timing analysis: fixed spatial (one region), as a time series
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg'
import numpy as np
import pandas as pd
import FDIP
import os
import draw_plot, map_plot
from importlib import reload
from matplotlib import pyplot as plt
import mannkendall_test
import variation_detect

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
season_params = pd.read_excel(os.path.join(home, "4.Analysis_timing/season_params.xlsx"), index_col=0)
Drought_year_number = pd.read_excel(os.path.join(home, "Drought_year_number.xlsx"), index_col=0)
FD_year_number = pd.read_excel(os.path.join(home, "FD_year_number.xlsx"), index_col=0)
Drought_year_number_Helong = pd.read_excel(os.path.join(home, "Drought_year_number_Helong.xlsx"), index_col=0)
Drought_year_number_noHelong = pd.read_excel(os.path.join(home, "Drought_year_number_noHelong.xlsx"), index_col=0)
FD_year_number_Helong = pd.read_excel(os.path.join(home, "FD_year_number_Helong.xlsx"), index_col=0)
FD_year_number_noHelong = pd.read_excel(os.path.join(home, "FD_year_number_noHelong.xlsx"), index_col=0)

mk_drought = np.loadtxt(os.path.join(home, "4.Analysis_timing/mk_drought.txt"), dtype=int)
mk_FD = np.loadtxt(os.path.join(home, "4.Analysis_timing/mk_FD.txt"), dtype=int)
slope_drought = np.loadtxt(os.path.join(home, "4.Analysis_timing/slope_drought.txt"), dtype=float)
slope_FD = np.loadtxt(os.path.join(home, "4.Analysis_timing/slope_FD.txt"), dtype=float)

year = np.arange(1948, 2015)
Num_point = 1166  # grid number

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
boundry_shpMap = map_plot.ShpMap(boundry_shp)
Helong_shpMap = map_plot.ShpMap(Helong_shp, linestyle="--", linewidth=0.5)

# data preprocessing
# spatial avg: sm_rz ~ time*point -> sm_rz ~ time (point avg)
sm_rz_pentad_spatial_avg = sm_rz_pentad.mean(axis=1)

# point mean: num ~ point*time -> num ~ time (point mean)
Drought_year_number_mean = Drought_year_number.mean(axis=0)
Drought_year_number_Helong_mean = Drought_year_number_Helong.mean(axis=0)
Drought_year_number_noHelong_mean = Drought_year_number_noHelong.mean(axis=0)
FD_year_number_mean = FD_year_number.mean(axis=0)
FD_year_number_Helong_mean = FD_year_number_Helong.mean(axis=0)
FD_year_number_noHelong_mean = FD_year_number_noHelong.mean(axis=0)


# time series of spaital average sm
def plot_sm():
    FD_avg = FDIP.FD(sm_rz_pentad_spatial_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5,
                     pc=0.28,
                     excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                     fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
    SM_percentile, RI, out_put, dp = FD_avg.general_out()
    # out_put.to_excel("out_put.xlsx")


# season analysis
facecolors = ['lightgreen', 'forestgreen', 'wheat', 'lightblue']


# boxplot of Drought season
def boxplot_drought_season():
    fig_boxplot_drought_season = draw_plot.Figure()
    draw_drought = draw_plot.Draw(fig_boxplot_drought_season.ax, fig_boxplot_drought_season, gridy=True,
                                  title="Boxplot of seasonal Drought number",
                                  labelx=None, labely="Number")
    box_drought = draw_plot.BoxDraw([season_params["Drought_spring"].values, season_params["Drought_summer"].values,
                                     season_params["Drought_autumn"].values, season_params["Drought_winter"].values],
                                    facecolors=facecolors, labels=["Spring", "Summer", "Autumn", "Winter"], notch=True,
                                    sym='r+', patch_artist=True, showfliers=False)
    draw_drought.adddraw(box_drought)


# boxplot of FD season
def boxplot_FD_season():
    fig_boxplot_FD_season = draw_plot.Figure()
    draw_FD = draw_plot.Draw(fig_boxplot_FD_season.ax, fig_boxplot_FD_season, gridy=True,
                             title="Boxplot of seasonal FD number",
                             labelx=None, labely="Number")
    box_FD = draw_plot.BoxDraw([season_params["FD_spring"].values, season_params["FD_summer"].values,
                                season_params["FD_autumn"].values, season_params["FD_winter"].values],
                               facecolors=facecolors, labels=["Spring", "Summer", "Autumn", "Winter"], notch=True,
                               sym='r+', patch_artist=True, showfliers=False)
    draw_FD.adddraw(box_FD)


# map of Drought season flag
def plot_drought_season_flag():
    fig_drought_season_flag = map_plot.Figure()
    map_drought_season_flag = map_plot.Map(fig_drought_season_flag.ax, fig_drought_season_flag, extent=extent,
                                           grid=True,
                                           res_grid=res_grid, res_label=res_label,
                                           title="Spatial Distribution of Drought Hot Season")
    # raster_drought_season_flag = map_plot.RasterMap_cb2(det=det, data_lat=lat, data_lon=lon,
    #                                                     data=season_params["Drought_season_Flag"].values,
    #                                                     expand=5, cb_label="Hot season")
    raster_drought_season_flag = map_plot.RasterMap_segmented_cb(colorlevel=[0, 1.5, 2.5, 3.5, 4.5],
                                            colordict=['lightgreen', 'forestgreen', 'wheat', 'lightblue'],
                                            cbticks=["Spring", "Summber", "Autumn", "Winter"],
                                                                 cbticks_position=[0.7, 1.7, 2.8, 3.9],
                                                                 det=det, data_lat=lat, data_lon=lon,
                                                        data=season_params["Drought_season_Flag"].values,
                                                        expand=5, cb_label="Hot season")
    map_drought_season_flag.addmap(raster_drought_season_flag)
    map_drought_season_flag.addmap(boundry_shpMap)


# map of FD season flag
def plot_FD_season_flag():
    fig_FD_season_flag = map_plot.Figure()
    map_FD_season_flag = map_plot.Map(fig_FD_season_flag.ax, fig_FD_season_flag, extent=extent, grid=True,
                                      res_grid=res_grid, res_label=res_label,
                                      title="Spatial Distribution of FD Hot Season")
    raster_drought_season_flag = map_plot.RasterMap_cb2(det=det, data_lat=lat, data_lon=lon,
                                                        data=season_params["FD_season_Flag"].values,
                                                        expand=5, cb_label="Hot season")
    map_FD_season_flag.addmap(raster_drought_season_flag)
    map_FD_season_flag.addmap(boundry_shpMap)


# MK Trend: Drought number and FD number - time series, point avg
def mktest_Drought_FD_number_mean():
    mk = mannkendall_test.MkTest(Drought_year_number_mean.values, x=year)
    mk.showRet(labelx="Year", labely="Number", title="MannKendall Test for Drought number")
    drought_mk = []
    drought_mk.append(mk.mkret)
    drought_mk.append(mk.senret)

    mk = mannkendall_test.MkTest(FD_year_number_mean.values, x=year)
    mk.showRet(labelx="Year", labely="Number", title="MannKendall Test for FD number")
    FD_mk = []
    FD_mk.append(mk.mkret)
    FD_mk.append(mk.senret)


# MK Trend: Drought number and FD number - map(significant)
def mktest_Drought_number_map():
    fig_mk_drought = map_plot.Figure()
    map_mk_drought = map_plot.Map(fig_mk_drought.ax, fig_mk_drought, extent=extent, grid=True,
                                  res_grid=res_grid, res_label=res_label,
                                  title="Spatial Distribution of MK Test for Drought Number")
    raster_mk_drought = map_plot.RasterMap_cb3(det=det, data_lat=lat, data_lon=lon,
                                               data=mk_drought, expand=5, cb_label="Trend")
    map_mk_drought.addmap(raster_mk_drought)
    map_mk_drought.addmap(boundry_shpMap)


def mktest_FD_number_map():
    fig_mk_FD = map_plot.Figure()
    map_mk_FD = map_plot.Map(fig_mk_FD.ax, fig_mk_FD, extent=extent, grid=True,
                             res_grid=res_grid, res_label=res_label,
                             title="Spatial Distribution of MK Test for FD Number")
    raster_mk_FD = map_plot.RasterMap_cb3(det=det, data_lat=lat, data_lon=lon,
                                          data=mk_FD, expand=5, cb_label="Trend")
    map_mk_FD.addmap(raster_mk_FD)
    map_mk_FD.addmap(boundry_shpMap)


# slope map for FD number
def mktest_slope_FD_number_map():
    fig_mk_slope_FD = map_plot.Figure()
    map_mk_slope_FD = map_plot.Map(fig_mk_slope_FD.ax, fig_mk_slope_FD, extent=extent, grid=True,
                                   res_grid=res_grid, res_label=res_label,
                                   title="Spatial Distribution of Sen'slope for FD Number")
    raster_mk_slope_FD = map_plot.RasterMap(det=det, data_lat=lat, data_lon=lon,
                                            data=slope_FD, expand=5, cb_label="Slope")  # , cmap_name="hot"
    map_mk_slope_FD.addmap(raster_mk_slope_FD)
    map_mk_slope_FD.addmap(boundry_shpMap)


# time_ticks set
time_ticks = {"ticks": year, "interval": 10}


# variation detect for Drought number - time series, point avg
def vd_detect_Drought_number():
    bgvd_drought_number = variation_detect.BGVD(Drought_year_number_mean.to_list())
    sccvd_drought_number = variation_detect.SCCVD(Drought_year_number_mean.to_list())
    mkvd_drought_number = variation_detect.MKVD(Drought_year_number_mean.to_list())
    ocvd_drought_number = variation_detect.OCVD(Drought_year_number_mean.to_list())
    bgvd_drought_number.plot(time_ticks=time_ticks)
    sccvd_drought_number.plot(time_ticks=time_ticks)
    mkvd_drought_number.plot(time_ticks=time_ticks)
    ocvd_drought_number.plot(time_ticks=time_ticks)


# variation detect for FD number - time series, point avg
def vd_detect_FD_number():
    bgvd_FD_number = variation_detect.BGVD(FD_year_number_mean.to_list())
    sccvd_FD_number = variation_detect.SCCVD(FD_year_number_mean.to_list())
    mkvd_FD_number = variation_detect.MKVD(FD_year_number_mean.to_list())
    ocvd_drought_number = variation_detect.OCVD(FD_year_number_mean.to_list())
    bgvd_FD_number.plot(time_ticks=time_ticks)
    sccvd_FD_number.plot(time_ticks=time_ticks)
    mkvd_FD_number.plot(time_ticks=time_ticks)
    ocvd_drought_number.plot(time_ticks=time_ticks)


if __name__ == '__main__':
    # boxplot_drought_season()
    # boxplot_FD_season()
    plot_drought_season_flag()
    # plot_FD_season_flag()
    # mktest_Drought_FD_number_mean()
    # mktest_Drought_number_map()
    # mktest_FD_number_map()
    # mktest_slope_FD_number_map()
    # vd_detect_Drought_number()
    # vd_detect_FD_number()
