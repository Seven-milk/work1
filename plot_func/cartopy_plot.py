# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# plot map with cartopy
from matplotlib import pyplot as plt
from cartopy import crs
from cartopy import feature
from cartopy.io.shapereader import Reader, natural_earth
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import os
import pandas as pd


def index_cal(extend: list, det: float, data_lat: np.ndarray, data_lon: np.ndarray):
    '''
    calculate index of the data_lat/lon in the extend
    input:
        extend: list extend = [lon_min, lon_max, lat_min, lat_max], is the center point
        det: resolution of a raster, unit = degree
        data_lat/lon: the lat/lon array of the data, 1D array, is the center point
    return:
        data_lat/lon_index index of the data_lat/lon in the extend, np.ndarray
    '''
    data_lat_index = np.array([int((data_lat[i] - extend[2]) / det) for i in range(len(data_lat))])
    data_lon_index = np.array([int((data_lon[i] - extend[0]) / det) for i in range(len(data_lon))])
    return data_lat_index, data_lon_index


def array_cal(extend: list, det: float, lat_index: np.ndarray, lon_index: np.ndarray, data: np.ndarray,
              maskvalue=-9999, expand=0):
    '''
    create full array based on the extend(a array), put the data into the full array, and mask the area without data
    input:
        extend: list extend = [lon_min, lon_max, lat_min, lat_max]
        det: resolution of a raster, unit = degree
        lat/lon_index: data_lat/lon_index index of the data_lat/lon in the extend, np.ndarray
        data: data, 1D array(correlated with lat/lon_index)
        expand: how many pixels used for expanding the array(for better plotting)
    output:
        array_data: full array with data, the area without data has been mask with maskvalue
        array_data_lon/lat: the lat/lon of the full array(array_data) calculated based on the extend, center point
    '''
    array_data = np.full(
        (int((extend[1] - extend[0]) / det + 1 + 2 * expand), int((extend[3] - extend[2]) / det + 1 + 2 * expand)),
        fill_value=maskvalue, dtype='float')
    mask = array_data == maskvalue
    array_data = np.ma.array(array_data, mask=mask)
    # put the data into the full array based on index
    for i in range(len(lat_index)):
        array_data[expand + lon_index[i], expand + lat_index[i]] = data[i]
    # array_data_lon/lat is the center point
    array_data_lon = np.linspace(extend[0] - det * expand, extend[1] + det * expand,
                                 num=int((extend[1] - extend[0]) / det + 1 + 2 * expand))
    array_data_lat = np.linspace(extend[2] - det * expand, extend[3] + det * expand,
                                 num=int((extend[3] - extend[2]) / det + 1 + 2 * expand))

    # move center point to edge, "pcolormesh" require *X* and *Y* can be used to specify the corners,
    # depend shading method, but the x/y(lat/lon) should specify the corners
    array_data_lon -= det / 2
    np.append(array_data_lon, array_data_lon[-1] + det)
    array_data_lat -= det / 2
    np.append(array_data_lat, array_data_lat[-1] + det)

    return array_data, array_data_lon, array_data_lat


def plot_cartopy_raster(extend: list, det: float, array_data: np.ndarray, array_data_lon: np.ndarray,
                        array_data_lat: np.ndarray, shape_file=None, proj=crs.PlateCarree(), cmap_name='YlOrBr',
                        dpi=300, grid=True, save=False, title="Map", cb_label="cb"):
    '''
    Using cartopy to plot raster map
    input:
        extend: the extend of data
        array_data: full array with data, the area without data has been mask with maskvalue
        array_data_lon/lat: the lat/lon of the full array(array_data) calculated based on the extend
        shape_file: list which save the shape_file path（.shp） to plot in the map, default = none(not plot)
        proj: projection, crs.Projection, it can be PlateCarree, AlbersEqualArea, AzimuthalEquidistant, EquidistantConic,
              LambertConformal, LambertCylindrical, Mercator, Miller, Mollweide, Orthographic, Robinson, Sinusoidal,
              Sinusoidal, TransverseMercator, UTM, InterruptedGoodeHomolosine...
              reference: https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
        cmap_name: it can be coolwarm
        dpi: figure dpi, default=300
        grid: bool, whether to open the grid lines
        save: bool, whether to save figure
        title: title and also is the figure name to save
    '''

    fig = plt.figure(dpi=dpi)
    ax = fig.subplots(1, 1, subplot_kw={"projection": proj})

    # add shape file of users'
    if shape_file is not None:
        for shape_path in shape_file:
            ax.add_feature(
                feature.ShapelyFeature(Reader(shape_path).geometries(), proj, edgecolor='k', facecolor='none'),
                linewidth=0.6, zorder=2)

    # add coastline with resolution = 50m
    ax.add_feature(feature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    # add River and lakes with resolution = 50m
    ax.add_feature(feature.RIVERS.with_scale('50m'), zorder=10)
    ax.add_feature(feature.LAKES.with_scale('50m'), zorder=10)

    # set extend_, is edge = center point +- det / 2
    extend_ = [min(array_data_lon) - det / 2, max(array_data_lon) + det / 2, min(array_data_lat) - det / 2,
               max(array_data_lat) + det / 2]
    ax.set_extent(extend_)

    # set gridlines
    if grid == True:
        gl = ax.gridlines(
            crs=proj,
            draw_labels=True,
            linewidth=0.2,
            color='k',
            alpha=0.5,
            linestyle='--'
        )
        gl.top_labels = False  # close top label
        gl.right_labels = False  # close right label
        gl.xformatter = LONGITUDE_FORMATTER  # set x label as lon format
        gl.yformatter = LATITUDE_FORMATTER  # set y label as lat format
        gl.xlocator = mticker.FixedLocator(np.arange(int(extend_[0]), int(extend_[1]) + 1, 1))
        gl.ylocator = mticker.FixedLocator(np.arange(int(extend_[2]), int(extend_[3]) + 1, 1))

    # general plot set
    font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
    font_ticks = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
    ax.set_title(title, fontsize=10, fontdict=font_title)
    cMap = plt.get_cmap(cmap_name)

    # plot raster
    pc = ax.pcolormesh(array_data_lon, array_data_lat, array_data.T, cmap=cMap)
    cb = plt.colorbar(pc, orientation='horizontal', extend='both', shrink=0.5)  # colorbar
    cb.ax.tick_params(labelsize=9)
    cb.set_label(cb_label, fontdict=font_label)
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
    plt.rcParams['font.size'] = 7
    plt.xticks(fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.show()

    if save == True:
        plt.savefig('./fig/' + title + '.jpg', dpi=350, bbox_inches='tight')
        plt.close()


def general_cartopy_plot(extend: list, det: float, data:np.ndarray, lat: np.ndarray, lon: np.ndarray, shape_file=None,
                         expand=10, grid=True, save=False, title="Map", cb_label="cb", cmap_name="YlOrBr"):
    """ general plot process: usually change in this function to spectify plot format """
    lat_index, lon_index = index_cal(extend=extend, det=det, data_lat=lat, data_lon=lon)
    array_data, array_data_lon, array_data_lat = array_cal(extend=extend, det=det, lat_index=lat_index,
                                                           lon_index=lon_index, data=data, expand=expand)
    plot_cartopy_raster(extend, det, array_data, array_data_lon, array_data_lat, shape_file=shape_file, proj=crs.PlateCarree(),
                        cmap_name=cmap_name, dpi=300, grid=grid, save=save, title=title, cb_label=cb_label)


if __name__ == "__main__":
    # general set
    root = "H"
    home = f"{root}:/research/flash_drough/"
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

    general_cartopy_plot(extend, det, sm_rz_time_avg, lat, lon)
