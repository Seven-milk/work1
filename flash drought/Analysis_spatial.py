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
import map_plot
import draw_plot

# general set
root = "H"
home = f"{root}:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment/SoilMoist_RZ_tavg.txt")
coord_path = os.path.join(home, "coord.txt")
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
sm_rz_Helong = np.loadtxt(os.path.join(home, "sm_rz_Helong.txt"), dtype="float", delimiter=" ")
sm_rz_noHelong = np.loadtxt(os.path.join(home, "sm_rz_noHelong.txt"), dtype="float", delimiter=" ")
date_pentad = np.loadtxt(os.path.join(home, "date_pentad.txt"), dtype="int")
sm_rz_pentad = np.loadtxt(os.path.join(home, "sm_rz_pentad.txt"))
sm_rz_Helong_pentad = np.loadtxt(os.path.join(home, "sm_rz_Helong_pentad.txt"))
sm_rz_noHelong_pentad = np.loadtxt(os.path.join(home, "sm_rz_noHelong_pentad.txt"))
sm_percentile_rz_pentad = np.loadtxt(os.path.join(home, "sm_percentile_rz_pentad.txt"), dtype="float", delimiter=" ")
Num_point = 1166  # grid number
static_params = pd.read_excel(os.path.join(home, "5.Analysis_spatial/static_params.xlsx"), index_col=0)
static_params_Helong = pd.read_excel(os.path.join(home, "5.Analysis_spatial/static_params_Helong.xlsx"), index_col=0)
static_params_noHelong = pd.read_excel(os.path.join(home, "5.Analysis_spatial/static_params_noHelong.xlsx"),
                                       index_col=0)
Drought_year_number = pd.read_excel(os.path.join(home, "Drought_year_number.xlsx"), index_col=0)
Drought_year_number_Helong = pd.read_excel(os.path.join(home, "Drought_year_number_Helong.xlsx"), index_col=0)
Drought_year_number_noHelong = pd.read_excel(os.path.join(home, "Drought_year_number_noHelong.xlsx"), index_col=0)
FD_year_number = pd.read_excel(os.path.join(home, "FD_year_number.xlsx"), index_col=0)
FD_year_number_Helong = pd.read_excel(os.path.join(home, "FD_year_number_Helong.xlsx"), index_col=0)
FD_year_number_noHelong = pd.read_excel(os.path.join(home, "FD_year_number_noHelong.xlsx"), index_col=0)
year = np.arange(1948, 2015)

# data preprocessing
# time avg: sm_rz ~ time*point -> sm_rz ~ point (time avg)
sm_rz_time_avg = sm_rz.mean(axis=0)
sm_rz_time_avg_Helong = sm_rz_Helong.mean(axis=0)
sm_rz_time_avg_noHelong = sm_rz_noHelong.mean(axis=0)

# point mean: num ~ point*time -> num ~ time (point mean)
Drought_year_number_mean = Drought_year_number.mean(axis=0)
Drought_year_number_Helong_mean = Drought_year_number_Helong.mean(axis=0)
Drought_year_number_noHelong_mean = Drought_year_number_noHelong.mean(axis=0)
FD_year_number_mean = FD_year_number.mean(axis=0)
FD_year_number_Helong_mean = FD_year_number_Helong.mean(axis=0)
FD_year_number_noHelong_mean = FD_year_number_noHelong.mean(axis=0)

# point flatten: sm_rz_pentad ~ time*point -> sm_rz_pentad ~ series
sm_rz_pentad_flatten = sm_rz_pentad.flatten()
sm_rz_Helong_pentad_flatten = sm_rz_Helong_pentad.flatten()
sm_rz_noHelong_pentad_flatten = sm_rz_noHelong_pentad.flatten()

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


# spatial plot of time average sm
def plot_sm():
    fig_sm = map_plot.Figure()
    map_sm = map_plot.Map(fig_sm.ax, fig_sm, extent=extent, grid=True, res_grid=res_grid, res_label=res_label,
                          title="Spatial Distribution of Time-avg Soil Moisture")
    raster_sm = map_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=sm_rz_time_avg,
                                   expand=5,
                                   cmap_name='RdBu', cb_label="Soil Moisture / mm")
    map_sm.addmap(raster_sm)
    # map_sm.addmap(Helong_shpMap)
    map_sm.addmap(boundry_shpMap)

    # save
    # fig_sm.save("1.Spatial Distribution of Time-avg Soil Moisture")


# spatial plot of Drought params
def plot_drought_params():
    fig_drought_params = map_plot.Figure(5)
    title = ["Drought FOC", "Drought number", "Mean Drought Duration", "Mean Drought Severity", "Mean SM_min"]
    cb_label = ["frequency", "number", "pentad", "pentad*\npercentile", "percentile"]
    data = [static_params["Drought_FOC"].values, static_params["Drought_number"].values,
            static_params["DD_mean"].values,
            static_params["DS_mean"].values, static_params["SM_min_mean"].values]
    map_boundry = [[0.35, 0.425], [10, 150], [10, 80], [1, 15], [0.05, 0.2]]
    for i in range(len(title)):
        map_ = map_plot.Map(fig_drought_params.ax[i], fig_drought_params, extent=extent, grid=True,
                            res_grid=res_grid,
                            res_label=res_label, title=title[i])
        map_.addmap(boundry_shpMap)
        # map_.addmap(Helong_shpMap)
        raster_ = map_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=data[i], expand=5,
                                     cmap_name='BrBG', cb_label=cb_label[i], map_boundry=map_boundry[i])
        map_.addmap(raster_)

    # save
    # fig_drought_params.save("2.spatial plot of Drought params")


# spatial plot of FD params
def plot_FD_params():
    fig_FD_params = map_plot.Figure(6)
    title = ["FD FOC", "FD number", "Mean FD Duration", "Mean FD Severity", "Mean RI_mean", "Mean RI_max"]
    cb_label = ["frequency", "number", "pentad", "pentad*\npercentile", "percentile/\npentad", "percentile/\npentad"]
    data = [static_params["FD_FOC"].values, static_params["FD_number"].values, static_params["FDD_mean"].values,
            static_params["FDS_mean"].values, static_params["RI_mean_mean"].values, static_params["RI_max_mean"].values]
    map_boundry = [[0.025, 0.125], [50, 150], [1, 4], [-0.3, -0.1], [0.025, 0.125], [0.05, 0.2]]
    for i in range(len(title)):
        map_ = map_plot.Map(fig_FD_params.ax[i], fig_FD_params, extent=extent, grid=True, res_grid=res_grid,
                            res_label=res_label, title=title[i])
        map_.addmap(boundry_shpMap)
        # map_.addmap(Helong_shpMap)
        raster_ = map_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=data[i], expand=5,
                                     cmap_name='BrBG', cb_label=cb_label[i], map_boundry=map_boundry[i])
        map_.addmap(raster_)

    # save
    # fig_FD_params.save("3.spatial plot of FD params")


# boxplot of Helong region / noHelong region / all region
x_label = ["total", "Helong", "noHelong"]
facecolors = ["lightgrey", 'lightgreen', 'lightblue']  # pink


def boxplot_sm():
    # sm mean
    fig_boxplot_sm = draw_plot.Figure()
    draw_sm = draw_plot.Draw(fig_boxplot_sm.ax, fig_boxplot_sm, gridy=True, title="Boxplot of Time-avg Soil Moisture",
                             labelx=None, labely="Percentile")
    box_sm = draw_plot.BoxDraw([sm_rz_time_avg, sm_rz_time_avg_Helong, sm_rz_time_avg_noHelong], facecolors=facecolors,
                               labels=x_label, notch=True, sym='r+', patch_artist=True, showfliers=False)
    draw_sm.adddraw(box_sm)

    # save
    # fig_boxplot_sm.save("4.boxplot_sm")


def boxplot_drought_params():
    # drought params
    fig_boxplot_drought_params = draw_plot.Figure(5)

    title = ["Drought FOC", "Drought number", "Mean Drought Duration", "Mean Drought Severity", "Mean SM_min"]
    y_label = ["frequency", "number", "pentad", "pentad*\npercentile", "percentile"]
    data = [static_params["Drought_FOC"].values, static_params["Drought_number"].values,
            static_params["DD_mean"].values,
            static_params["DS_mean"].values, static_params["SM_min_mean"].values]
    data_Helong = [static_params_Helong["Drought_FOC"].values, static_params_Helong["Drought_number"].values,
                   static_params_Helong["DD_mean"].values,
                   static_params_Helong["DS_mean"].values, static_params_Helong["SM_min_mean"].values]
    data_noHelong = [static_params_noHelong["Drought_FOC"].values, static_params_noHelong["Drought_number"].values,
                     static_params_noHelong["DD_mean"].values,
                     static_params_noHelong["DS_mean"].values, static_params_noHelong["SM_min_mean"].values]

    for i in range(len(title)):
        draw_ = draw_plot.Draw(fig_boxplot_drought_params.ax[i], fig_boxplot_drought_params, gridy=True,
                               title=title[i], labelx=None, labely=y_label[i])
        box_ = draw_plot.BoxDraw([data[i], data_Helong[i], data_noHelong[i]], facecolors=facecolors,
                                 labels=x_label, notch=True, sym='k+', patch_artist=True, showfliers=False)
        draw_.adddraw(box_)

    # save
    # fig_boxplot_drought_params.save("5.fig_boxplot_drought_params")


def boxplot_FD_params():
    # drought params
    fig_boxplot_FD_params = draw_plot.Figure(6)

    title = ["FD FOC", "FD number", "Mean FD Duration", "Mean FD Severity", "Mean RI_mean", "Mean RI_max"]
    y_label = ["frequency", "number", "pentad", "pentad*\npercentile", "percentile/pentad", "percentile/pentad"]
    data = [static_params["FD_FOC"].values, static_params["FD_number"].values, static_params["FDD_mean"].values,
            static_params["FDS_mean"].values, static_params["RI_mean_mean"].values, static_params["RI_max_mean"].values]
    data_Helong = [static_params_Helong["FD_FOC"].values, static_params_Helong["FD_number"].values,
                   static_params_Helong["FDD_mean"].values,
                   static_params_Helong["FDS_mean"].values, static_params_Helong["RI_mean_mean"].values,
                   static_params_Helong["RI_max_mean"].values]
    data_noHelong = [static_params_noHelong["FD_FOC"].values, static_params_noHelong["FD_number"].values,
                     static_params_noHelong["FDD_mean"].values,
                     static_params_noHelong["FDS_mean"].values, static_params_noHelong["RI_mean_mean"].values,
                     static_params_noHelong["RI_max_mean"].values]

    for i in range(len(title)):
        draw_ = draw_plot.Draw(fig_boxplot_FD_params.ax[i], fig_boxplot_FD_params, gridy=True,
                               title=title[i], labelx=None, labely=y_label[i])
        box_ = draw_plot.BoxDraw([data[i], data_Helong[i], data_noHelong[i]], facecolors=facecolors,
                                 labels=x_label, notch=True, sym='k+', patch_artist=True, showfliers=False)
        draw_.adddraw(box_)

    # save
    # fig_boxplot_FD_params.save("6.fig_boxplot_FD_params")


# scatter plot of Helong region / noHelong region / all region
def scatterplot_Drought_FD_character():
    fig_sactter_Drought = draw_plot.Figure(4)
    title = ["Drought Characteristic", "FD Characteristic", "Drought number", "FD number"]
    labels = ["total", "Helong", "noHelong"]  # legend
    x_label = ["Duration", "Duration", "Date", "Date"]
    y_label = ["Severity", "Severity", "Number", "Number"]
    data_x = [static_params["DD_mean"].values, static_params["FDD_mean"].values, year, year]
    data_y = [static_params["DS_mean"].values, static_params["FDS_mean"].values, Drought_year_number_mean,
              FD_year_number_mean]
    data_x_Helong = [static_params_Helong["DD_mean"].values, static_params_Helong["FDD_mean"].values, year, year]
    data_y_Helong = [static_params_Helong["DS_mean"].values, static_params_Helong["FDS_mean"].values,
                     Drought_year_number_Helong_mean, FD_year_number_Helong_mean]
    data_x_noHelong = [static_params_noHelong["DD_mean"].values, static_params_noHelong["FDD_mean"].values, year, year]
    data_y_noHelong = [static_params_noHelong["DS_mean"].values, static_params_noHelong["FDS_mean"].values,
                       Drought_year_number_noHelong_mean, FD_year_number_noHelong_mean]

    for i in range(len(title)):
        draw_ = draw_plot.Draw(fig_sactter_Drought.ax[i], fig_sactter_Drought, title=title[i], labelx=x_label[i],
                               labely=y_label[i], legend_on={"loc": "upper right", "framealpha": 0.8})
        # scatter_ = draw_plot.ScatterDraw(data_x[i], data_y[i], label=labels[0], color="grey", s=2, marker=".", zorder=20)
        scatter_Helong = draw_plot.ScatterDraw(data_x_Helong[i], data_y_Helong[i], label=labels[1], color="green",
                                               s=3, marker="^", zorder=25)
        scatter_noHelong = draw_plot.ScatterDraw(data_x_noHelong[i], data_y_noHelong[i], label=labels[2],
                                                 color="b", s=3, marker="^", zorder=20)
        # draw_.adddraw(scatter_)
        draw_.adddraw(scatter_Helong)
        draw_.adddraw(scatter_noHelong)

    # save
    # fig_sactter_Drought.save("7.scatterplot_Drought_FD_character")


# hist plot for sm_rz_pentad in Helong & noHelong
def histplot_sm_rz_pentad_faltten():
    fig_hist_sm = draw_plot.Figure()
    draw_hist_sm = draw_plot.Draw(fig_hist_sm.ax, fig_hist_sm, title="Soil Moisture Hist", labelx="SM / mm",
                                  labely="Number", legend_on=True)
    hist_ = draw_plot.HistDraw(sm_rz_pentad_flatten, label="total", alpha=0.5, color="grey", bins=100)
    hist_noHelong = draw_plot.HistDraw(sm_rz_noHelong_pentad_flatten, label="noHelong", alpha=0.5, color="blue",
                                       bins=100)
    hist_Helong = draw_plot.HistDraw(sm_rz_Helong_pentad_flatten, label="Helong", alpha=1, color="green", bins=100)
    draw_hist_sm.adddraw(hist_)
    draw_hist_sm.adddraw(hist_noHelong)
    draw_hist_sm.adddraw(hist_Helong)

    # save
    # fig_hist_sm.save("8.histplot_sm_rz_pentad_faltten")


if __name__ == "__main__":
    plot_sm()
    plot_drought_params()
    plot_FD_params()
    boxplot_sm()
    boxplot_drought_params()
    boxplot_FD_params()
    scatterplot_Drought_FD_character()
    histplot_sm_rz_pentad_faltten()
