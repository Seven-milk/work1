# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# plot map with cartopy
from matplotlib import pyplot as plt
from cartopy import crs
from cartopy import feature
from cartopy.io.shapereader import Reader, natural_earth
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib
from importlib import reload
import cv2
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import os
import pandas as pd
import abc
from rasterio.plot import show
import rasterio

''' usage 

f = Figure()
m = Map(f.ax, f, grid=True, res_grid=1, res_label=3)
m.addmap(BaseMap())
r = RasterMap(extend, det, lat, lon, sm_rz_time_avg, expand=5)
m.addmap(r)
s = ShpMap(shape_file, proj=crs.PlateCarree())
m.addmap(s)
f.fig.show()

    stracture
----------------------------------------------------------------------------------------------------------------------
|      ---  MapBase(abstract class) --- ——> plot --------------------------------------------                        |
|      |            |             |                                                         |                        |
|    BaseMap    RasterMap      ShapeMap [——> plot] (concrete class)                         |                        |
|                                                                                           |                        |
|   MeshgridArray(base class) [trans 1D array into 2D array]---> RasterMap                  |                        |
|                                                                                           V                        |
|   Figure --- plot multifigure ---> ax[i] ---> Map ---> addMap(add multimap in one ax) [Map class, Map.plot()]      |
|    |               |             Figure.fig    |                                                                   |
|   .reset()     .addfig()                     .set()                                                                |
|   .save()          |                                                                                               |
|                .calrowcol()                                                                                        |
----------------------------------------------------------------------------------------------------------------------
'''


# Define MapBase class
class MapBase(abc.ABC):
    ''' MapBase abstract class '''

    @abc.abstractmethod
    def plot(self, ax, Fig):
        ''' plot map '''


class MeshgridArray:
    ''' This class meshes a original data (1D array) into a full data (2D array) based on a extent, det, data, data_lat,
        data_lon '''

    def __init__(self, det: float, data_lat: np.ndarray, data_lon: np.ndarray, data: np.ndarray,
                 maskvalue=-9999, expand=0):
        ''' init function
        input:
            det: resolution of a raster, unit = degree
            data_lat/lon: the lat/lon array of the data, 1D array, is the center point
            data: data, 1D array(correlated with lat/lon_index)
            expand: how many pixels used for expanding the array(for better plotting)
        Main output:
            self.array_data: full array with data, the area without data has been mask with maskvalue
            self.array_data_lon/lat: the lat/lon of the full array(array_data) calculated based on the extent, center point
        '''
        # load data
        self.extent = [min(data_lon), max(data_lon), min(data_lat), max(data_lat)]
        self.det = det
        self.data_lat = data_lat
        self.data_lon = data_lon
        self.data = data
        self.maskvalue = maskvalue
        self.expand = expand

        self.lat_index, self.lon_index = self.index_cal()  # calculate index of the data_lat/lon in the extent

        self.array_data, self.array_data_lon, self.array_data_lat = self.array_cal()  # create full array based on the
        # extent(a array), put the data into the full array, and mask the area without data

    def index_cal(self):
        '''
        calculate index of the data_lat/lon in the extent
        input:
            extent: list extent = [lon_min, lon_max, lat_min, lat_max], is the center point
            det: resolution of a raster, unit = degree
            data_lat/lon: the lat/lon array of the data, 1D array, is the center point
        return:
            data_lat/lon_index index of the data_lat/lon in the extent, np.ndarray
        '''
        data_lat_index = np.array(
            [int((self.data_lat[i] - self.extent[2]) / self.det) for i in range(len(self.data_lat))])
        data_lon_index = np.array(
            [int((self.data_lon[i] - self.extent[0]) / self.det) for i in range(len(self.data_lon))])
        return data_lat_index, data_lon_index

    def array_cal(self):
        '''
        create full array based on the extent(a array), put the data into the full array, and mask the area without data
        input:
            extent: list extent = [lon_min, lon_max, lat_min, lat_max]
            det: resolution of a raster, unit = degree
            lat/lon_index: data_lat/lon_index index of the data_lat/lon in the extent, np.ndarray
            data: data, 1D array(correlated with lat/lon_index)
            expand: how many pixels used for expanding the array(for better plotting)
        output:
            array_data: full array with data, the area without data has been mask with maskvalue
            array_data_lon/lat: the lat/lon of the full array(array_data) calculated based on the extent, center point
        '''
        array_data = np.full(
            (int((self.extent[1] - self.extent[0]) / self.det + 1 + 2 * self.expand),
             int((self.extent[3] - self.extent[2]) / self.det + 1 + 2 * self.expand)),
            fill_value=self.maskvalue, dtype='float')
        mask = array_data == self.maskvalue
        array_data = np.ma.array(array_data, mask=mask)
        # put the data into the full array based on index
        for i in range(len(self.lat_index)):
            array_data[self.expand + self.lon_index[i], self.expand + self.lat_index[i]] = self.data[i]
        # array_data_lon/lat is the center point
        array_data_lon = np.linspace(self.extent[0] - self.det * self.expand, self.extent[1] + self.det * self.expand,
                                     num=int((self.extent[1] - self.extent[0]) / self.det + 1 + 2 * self.expand))
        array_data_lat = np.linspace(self.extent[2] - self.det * self.expand, self.extent[3] + self.det * self.expand,
                                     num=int((self.extent[3] - self.extent[2]) / self.det + 1 + 2 * self.expand))

        # move center point to edge, "pcolormesh" require *X* and *Y* can be used to specify the corners,
        # depend shading method, but the x/y(lat/lon) should specify the corners(based on different env the raster might
        # out of alignment, to correct, use follow code)
        # array_data_lon -= self.det / 2
        # np.append(array_data_lon, array_data_lon[-1] + self.det)
        # array_data_lat -= self.det / 2
        # np.append(array_data_lat, array_data_lat[-1] + self.det)

        return array_data, array_data_lon, array_data_lat


class BaseMap(MapBase):
    ''' base map '''

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        # add coastline with resolution = 50m
        ax.add_feature(feature.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
        # add River and lakes with resolution = 50m
        ax.add_feature(feature.RIVERS.with_scale('50m'), zorder=10)
        ax.add_feature(feature.LAKES.with_scale('50m'), zorder=10)


class RasterMap(MeshgridArray, MapBase):
    ''' raster map '''

    def __init__(self, det: float, data_lat: np.ndarray, data_lon: np.ndarray, data: np.ndarray,
                 maskvalue=-9999, expand: int = 0, cmap_name='RdBu', map_boundry=None, cb_label="cb"):
        ''' init function
        input:
            det: resolution of a raster, unit = degree
            data_lat/lon: the lat/lon array of the data, 1D array, is the center point
            data: data, 1D array(correlated with lat/lon_index)
            expand: how many pixels used for expanding the array(for better plotting)
            cmap_name: cmap name, BrBG, RdBu, seismic, bwr, coolwarm, afmhot, twilight_shifted; suffix _r is inversion
            map_boundry: none or list, if list, this param sets the map/colorbar boundry - [vmin, vmax]
            cb_label=string, the label of colorbar, if none, do not plot cb
        '''
        MeshgridArray.__init__(self, det, data_lat, data_lon, data, maskvalue, expand)
        self.cmap_name = cmap_name
        self.map_boundry = map_boundry
        self.cb_label = cb_label
        self.extent_plot = [min(self.array_data_lon) - self.det / 2, max(self.array_data_lon) + self.det / 2,
                            min(self.array_data_lat) - self.det / 2, max(self.array_data_lat) + self.det / 2]

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        ax.set_extent(self.extent_plot)
        cMap = plt.get_cmap(self.cmap_name)
        # plot raster
        if self.map_boundry == None:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               norm=mcolors.Normalize(clip=True))
        else:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               vmin=self.map_boundry[0], vmax=self.map_boundry[1], norm=mcolors.Normalize(clip=True))

        # cb
        if self.cb_label != None:
            shrinkrate = 0.7 if isinstance(Fig.ax, np.ndarray) else 0.9
            extend = 'neither' if isinstance(Fig.ax, np.ndarray) else 'both'
            cb = Fig.fig.colorbar(pc, ax=ax, orientation='vertical', shrink=shrinkrate, pad=0.01, extend=extend)
            cb.ax.tick_params(labelsize=Fig.font_label["size"], direction='in')
            if isinstance(Fig.ax, np.ndarray):
                cb.ax.set_title(label=self.cb_label, fontdict=Fig.font_label)
            else:
                cb.set_label(self.cb_label, fontdict=Fig.font_label)
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_family('Arial')


class RasterMap_segmented_cb(RasterMap):
    ''' raster map with segmented colorbar '''

    def __init__(self, colorlevel, colordict, cbticks, cbticks_position, det: float, data_lat: np.ndarray,
                 data_lon: np.ndarray, data: np.ndarray, maskvalue=-9999, expand: int = 0, cmap_name='RdBu',
                 map_boundry=None, cb_label="cb"):
        ''' init function
        input:
            similar to RasterMap
            colorlevel: list, colorlevel for colorbar
            colordict: list, contains color string
            cbticks: list, contains colorbar ticks
            cbticks_position: list, the ticks position, need to adjust by your self, default = middle numbers in
                              colorlevel
            cb_label=string, the label of colorbar, if none, do not plot cb

            i.e.
            colorlevel: [0, 1.5, 2.5, 3.5, 4.5]
            colordict: ['lightgreen', 'forestgreen', 'wheat', 'lightblue']
            cbticks: ["Spring", "Summber", "Autumn", "Winter"]
            cbticks_position: [0.5, 2.0, 3.0, 4.0] -> [0.7, 1.7, 2.8, 3.9]

        '''

        super(RasterMap_segmented_cb, self).__init__(det, data_lat, data_lon, data, maskvalue, expand, cmap_name,
                                                     map_boundry, cb_label)
        self.colorlevel = colorlevel
        self.colordict = colordict
        self.cbticks = cbticks
        self.cbticks_position = cbticks_position

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        # cb set
        colorlevel = self.colorlevel
        colordict = self.colordict
        ticks = self.cbticks
        ticks_position = self.cbticks_position

        cmap = mcolors.ListedColormap(colordict)
        norm = mcolors.BoundaryNorm(colorlevel, cmap.N)

        # other set
        ax.set_extent(self.extent_plot)

        # plot raster
        cMap = cmap
        if self.map_boundry == None:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               norm=norm)
        else:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               vmin=self.map_boundry[0], vmax=self.map_boundry[1], norm=norm)

        # cb
        if self.cb_label != None:
            extend = "neither"

            cb = Fig.fig.colorbar(pc, ax=ax, orientation='vertical', pad=0.01, extend=extend)
            cb.ax.yaxis.set_major_locator(plt.NullLocator())

            cb2_ax = cb.ax.secondary_yaxis('right')
            cb2_ax.tick_params(labelsize=Fig.font_label["size"], direction='in')
            cb2_ax.set_yticks(ticks_position)
            cb2_ax.set_yticklabels(ticks)
            cb.ax.set_title(label=self.cb_label, fontdict=Fig.font_label)

            for l in cb2_ax.yaxis.get_ticklabels():
                l.set_family('Arial')


class RasterMap_cb2(RasterMap):
    ''' raster map with cb2 '''

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        # cb
        colorlevel = [0, 1.5, 2.5, 3.5, 4.5]
        colordict = ['lightgreen', 'forestgreen', 'wheat', 'lightblue']
        tick = ["Spring", "Summber", "Autumn", "Winter"]
        cmap = mcolors.ListedColormap(colordict)
        norm = mcolors.BoundaryNorm(colorlevel, cmap.N)

        ax.set_extent(self.extent_plot)
        cMap = cmap
        # plot raster
        if self.map_boundry == None:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               norm=norm)
        else:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               vmin=self.map_boundry[0], vmax=self.map_boundry[1], norm=norm)

        # shrinkrate = 0.7 if isinstance(Fig.ax, np.ndarray) else 0.9
        extend = 'neither'

        cb = Fig.fig.colorbar(pc, ax=ax, orientation='vertical', pad=0.01, extend=extend)
        cb.ax.yaxis.set_major_locator(plt.NullLocator())

        cb2_ax = cb.ax.secondary_yaxis('right')
        cb2_ax.tick_params(labelsize=Fig.font_label["size"], direction='in')
        cb2_ax.set_yticks([0.7, 1.7, 2.8, 3.9])
        cb2_ax.set_yticklabels(tick)
        cb.ax.set_title(label=self.cb_label, fontdict=Fig.font_label)

        for l in cb2_ax.yaxis.get_ticklabels():
            l.set_family('Arial')


class RasterMap_cb3(RasterMap):
    ''' raster map with cb2 '''

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        # cb
        colorlevel = [-1.5, -0.5, 0.5, 1.5]
        colordict = ['green', 'lightgrey', 'red']
        tick = ["downtrend", "no trend", "uptrend"]
        cmap = mcolors.ListedColormap(colordict)
        norm = mcolors.BoundaryNorm(colorlevel, cmap.N)

        ax.set_extent(self.extent_plot)
        cMap = cmap
        # plot raster
        if self.map_boundry == None:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               norm=norm)
        else:
            pc = ax.pcolormesh(self.array_data_lon, self.array_data_lat, self.array_data.T, cmap=cMap,
                               vmin=self.map_boundry[0], vmax=self.map_boundry[1], norm=norm)

        # shrinkrate = 0.7 if isinstance(Fig.ax, np.ndarray) else 0.9
        extend = 'neither'

        cb = Fig.fig.colorbar(pc, ax=ax, orientation='vertical', pad=0.01, extend=extend)
        cb.ax.yaxis.set_major_locator(plt.NullLocator())

        cb2_ax = cb.ax.secondary_yaxis('right')
        cb2_ax.tick_params(labelsize=Fig.font_label["size"], direction='in')
        cb2_ax.set_yticks([-1, 0, 1])
        cb2_ax.set_yticklabels(tick)
        cb.ax.set_title(label=self.cb_label, fontdict=Fig.font_label)

        for l in cb2_ax.yaxis.get_ticklabels():
            l.set_family('Arial')


class ShpMap(MapBase):
    ''' shp map '''

    def __init__(self, shape_file: list = None, proj: crs.Projection = crs.PlateCarree(), edgecolor: str = "k",
                 facecolor: str = "none",
                 **kwargs):
        ''' init function
        input:
            shape_file: list of str, which save the shape_file path（.shp） to plot in the map, default = none(not plot)
            proj: crs.Projection, projection
            edgecolor: str, the edgecolor of this shp
            facecolor: str, the facecolor of this shp
            **kwargs: keyword args, it could contain "linewidth", "zorder", "linestyle" "alpha"
        '''
        self.shape_file = shape_file
        self.proj = proj
        self.edgecolor = edgecolor
        self.facecolor = facecolor
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        # add shape file of users'
        for shape_path in self.shape_file:
            ax.add_feature(feature.ShapelyFeature(Reader(shape_path).geometries(), crs=self.proj,
                                                  edgecolor=self.edgecolor, facecolor=self.facecolor), **self.kwargs)


class ImgMap(MapBase):
    ''' Img Map '''
    def __init__(self, img_file: list = None, proj: crs.Projection = crs.PlateCarree(), flags=-1,
                 **kwargs):
        ''' init function 
        input:
            img_file: list of str, which save the img_file path（.tiff, ） to plot in the map, default = none(not plot)
            proj: crs.Projection, projection
            flags: flags in cv2.imread
                IMREAD_UNCHANGED = -1
                IMREAD_GRAYSCALE = 0
                IMREAD_COLOR = 1
                IMREAD_ANYDEPTH = 2
                IMREAD_ANYCOLOR = 4

            support:
            .   -   Windows bitmaps - \*.bmp, \*.dib (always supported)
            .   -   JPEG files - \*.jpeg, \*.jpg, \*.jpe (see the *Note* section)
            .   -   JPEG 2000 files - \*.jp2 (see the *Note* section)
            .   -   Portable Network Graphics - \*.png (see the *Note* section)
            .   -   WebP - \*.webp (see the *Note* section)
            .   -   Portable image format - \*.pbm, \*.pgm, \*.ppm \*.pxm, \*.pnm (always supported)
            .   -   PFM files - \*.pfm (see the *Note* section)
            .   -   Sun rasters - \*.sr, \*.ras (always supported)
            .   -   TIFF files - \*.tiff, \*.tif (see the *Note* section)
            .   -   OpenEXR Image files - \*.exr (see the *Note* section)
            .   -   Radiance HDR - \*.hdr, \*.pic (always supported)
            .   -   Raster and Vector geospatial data supported by GDAL (see the *Note* section)

            **kwargs: keyword args in ax.imshow, it could contain "origin", "regrid_shape", "extent"
                    extend: list, The corner coordinates of the image in the form ``(left, right, bottom, top)``.
                    The coordinates should be in the coordinate system passed to the transform keyword.
        '''
        self._img_file = img_file
        self._proj = proj
        self._kwargs = kwargs
        self._flags = flags
        
    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        for img_path in self._img_file:
            img_ = cv2.imread(img_path, self._flags)  # read as bgr
            trans_img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)  # trans into rgb plotting in ax.imshow()
            ax.imshow(
                trans_img,
                transform=self._proj,
                **self._kwargs
            )


class TiffMap(MapBase):
    ''' Tiff Map '''
    def __init__(self, tif_file: list = None, **kwargs):
        ''' init function
        input:
            tif_file: list of str, which save the tif_file path（.tiff） to plot in the map
            proj: crs.Projection, projection
            *kwargs: keyword args, it could contain "with_bounds", "contour", "contour_label_kws", "title", "transform",
                    "adjust", see rasterio.plot.show()
        '''
        self._tif_file = tif_file
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        # add tiff file of users'
        for tif_path in self._tif_file:
            rf = rasterio.open(tif_path, mode="r")
            show(rf, ax=ax, **self.kwargs)
        ax.set_extent([rf.bounds.left, rf.bounds.right, rf.bounds.bottom, rf.bounds.top])


class TextMap(MapBase):
    ''' Text Map '''

    def __init__(self, text: str, extent: list, **kwargs):
        ''' init function
        input:
            text: str, the text to plot
            extent: list of two elements, [lon/x, lat/y], define the position to plot text
            kwargs: keyword args, it could contain "color" "fontdict"(dict) "alpha" "zorder" ...
        '''
        self.text = text
        self.extent = extent
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implement the MapBase.plot function '''
        # define the default fontdict
        if "fontdict" not in self.kwargs.keys():
            self.kwargs["fontdict"] = Fig.font_label
        ax.text(self.extent[0], self.extent[1], self.text, **self.kwargs)


class Figure:
    ''' figure set '''

    def __init__(self, addnumber: int = 1, dpi: int = 300, figsize=(12, 5), proj: crs.Projection = crs.PlateCarree(),
                 wspace=None, hspace=None, family="Arial", figRow=1, figCol=1, axflatten=True, **kwargs):
        ''' init function
        input:
            addnumber: the init add fig number
            dpi: figure dpi, default=300
            figsize: figure size, default=(12, 5)
            proj: the init proj for each subplots, crs.Projection, it can be crs.: PlateCarree, AlbersEqualArea,
                  AzimuthalEquidistant, EquidistantConic, LambertConformal, LambertCylindrical, Mercator, Miller,
                  Mollweide, Orthographic, Robinson, Sinusoidal, Sinusoidal, TransverseMercator, UTM,
                  InterruptedGoodeHomolosine...
                  reference: https://scitools.org.uk/cartopy/docs/latest/crs/projections.html
            wspace/hspace: the space between subfig
            family: font family
            figRow=1, figCol=1: to set the fig row and col
            axflatten: whether flatten the ax
            **kwargs: keyword args of subplots, it could contain "sharex" "sharey"

        self.figNumber: fig number in the base map, default=1
        self.figRow: the row of subfigure, default=1
        self.figCol: the col of subfigure, default=1

        Main output: self.ax, a list of subfig in a canvas used to plot
        note:
        1) if the fig close, call Figure.fig.show() function
        2) if figNumber is odd by addfig, it will be set to even by self.calrowcol()
        3) if figNumber==1, dont use Figure.ax[0], just use Figure.ax
        '''
        self.figNumber = 0
        self.figRow = figRow
        self.figCol = figCol
        self.dpi = dpi
        self.figsize = figsize
        self.kwargs = kwargs
        self.fig = plt.figure(dpi=self.dpi, figsize=self.figsize)
        self.proj = proj
        self.add = False
        self.axflatten = axflatten
        self.wspace = wspace
        self.hspace = hspace
        self.addFig(addnumber, wspace=self.wspace, hspace=self.hspace, **self.kwargs)
        self.font_label = {'family': family, 'weight': 'normal',
                           'size': 6 if isinstance(self.ax, np.ndarray) else 8}
        self.font_ticks = {'family': family, 'weight': 'normal',
                           'size': 6 if isinstance(self.ax, np.ndarray) else 8}
        self.font_title = {'family': family, 'weight': 'bold',
                           'size': 6 if isinstance(self.ax, np.ndarray) else 8}
        if self.add == True:
            self.unview_last()

    def addFig(self, AddNumber=1, wspace=0.2, hspace=0.1, **kwargs):
        ''' add blank figure and return ax '''
        self.figNumber += AddNumber
        if self.figNumber >= 2:
            self.calrowcol()
        self.fig.clf()
        self.ax = self.fig.subplots(nrows=self.figRow, ncols=self.figCol, subplot_kw={"projection": self.proj},
                                    **kwargs)
        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
        if isinstance(self.ax, np.ndarray) and self.axflatten:
            self.ax = self.ax.flatten()

    def calrowcol(self, rowfirst=True):
        ''' Decomposition factor of self.figNumber to get self.figRow and self.figCol
            rowfirst: row first calculating
        '''
        # if self.figNumber == 2
        if self.figNumber == 2:
            self.figRow = 2
            self.figCol = 1
            if rowfirst == False:
                self.figRow, self.figCol = self.figCol, self.figRow
            return

        # Determine if self.figNumber is prime and decomposition it
        while True:
            # prime
            for i in range(2, self.figNumber):
                if not self.figNumber % i:  # remainder equal to 0
                    self.figRow = i
                    self.figCol = self.figNumber // self.figRow
                    if rowfirst == False:
                        self.figRow, self.figCol = self.figCol, self.figRow
                    return
            # not prime: self.figNumber + 1 (blank subplot)
            self.figNumber += 1
            self.add = True

    def unview_last(self):
        ''' unview the last ax '''
        self.ax[-1].set_visible(False)

    def reset(self):
        ''' reset Figure to the init state, the canvas is still exist that can be used '''
        self.fig.clf()
        self.figNumber = 0
        self.figRow = 1
        self.figCol = 1
        self.addFig()

    def resetax(self, num=0, colorbar_num=0):
        ''' reset ax to the init state, num start from 0, the ax is still exist that can be used '''
        ax_ = [ax for ax in self.fig.get_axes() if ax._label != "<colorbar>"]
        ax_[num].cla()
        ax_[num].outline_patch.set_visible(False)
        ax_bar = [ax for ax in self.fig.get_axes() if ax._label == "<colorbar>"]
        if len(ax_bar) != 0:
            ax_bar[colorbar_num].remove()

    def save(self, title):
        ''' save fig
        input:
            title: the title to save figure
        '''
        if not os.path.exists(os.path.join(os.getcwd(), 'fig')):
            os.mkdir(os.path.join(os.getcwd(), 'fig'))
        plt.savefig('fig/' + title + '.jpg', dpi=self.dpi, bbox_inches='tight')

    def show(self):
        ''' show fig '''
        plt.show()


class Map:
    ''' Add map(has proj) in one ax(Geoax), this class is used to represent ax and plot map '''

    def __init__(self, ax, Fig: Figure, extent=None, proj: crs.Projection = crs.PlateCarree(),
                 grid=False, res_grid=5, res_label=5, axoff=False, title="map"):
        ''' init function
        input:
            ax: a single ax for this map from Figure.ax[i]
            fig: Figure, the Figure.fig contain this ax, implement the communication between Map and Fig (for plot colobar)
            proj: crs.Projection, the projection for this ax
            extent: list extent = [lon_min, lon_max, lat_min, lat_max], is the center point, used to define plot boundry
            proj: proj: projection for this map, crs.Projection
            grid: bool, whether to open the grid lines
            res_grid: resolution for grid
            title: title of this ax
        '''
        self.ax = ax
        self.Fig = Fig
        self.extent = extent
        self.proj = proj
        self.grid = grid
        self.res_grid = res_grid
        self.res_label = res_label
        self.title = title
        self.set(self.extent, self.proj, self.grid, self.res_grid, self.res_label, self.title)
        if axoff == True:
            self.axoff()

    def addmap(self, map: MapBase):
        ''' add map
        input:
            map: MapBase class, it can be the sub class of MapBase: such as BaseMap, RasterMap, ShpMap...
        '''
        map.plot(self.ax, self.Fig)

    def set(self, extent=None, proj: crs.Projection = crs.PlateCarree(), grid=False, res_grid=5, res_label=5,
            title="map"):
        ''' set this Map(ax)
        input:
            extent: list extent = [lon_min, lon_max, lat_min, lat_max], is the center point
            proj: proj: projection for this map, crs.Projection
            grid: bool, whether to open the grid lines
            res_grid: resolution for grid
            res_label: resolution for grid label, it should be n*res_grid
            title: title of this ax
        '''
        if extent == None:
            extent = [-180, 180, -90, 90]
        # set extent and proj
        self.ax.set_extent(extent, crs=proj)
        # set gridlines
        if grid == True:
            # grid without label (sub-grid)
            gl2 = self.ax.gridlines(
                crs=proj,
                draw_labels=False,
                linewidth=0.2,
                color='k',
                alpha=0.5,
                linestyle='--',
                auto_inline=True
            )
            gl2.xlocator = mticker.FixedLocator(
                np.arange(int(extent[0]) - 5 * res_grid, int(extent[1]) + 5 * res_grid, res_grid))  # set grid label
            gl2.ylocator = mticker.FixedLocator(
                np.arange(int(extent[2]) - 5 * res_grid, int(extent[3]) + 5 * res_grid, res_grid))

            # label grid (The parent grid)
            gl = self.ax.gridlines(
                crs=proj,
                draw_labels=True,
                linewidth=0.2,
                color='k',
                alpha=0.5,
                linestyle='--',
                auto_inline=True
            )
            gl.top_labels = False  # close top label
            gl.right_labels = False  # close right label
            gl.xformatter = LONGITUDE_FORMATTER  # set x label as lon format
            gl.yformatter = LATITUDE_FORMATTER  # set y label as lat format
            gl.xlocator = mticker.FixedLocator(
                np.arange(int(extent[0]) - 5 * res_grid, int(extent[1]) + 5 * res_grid, res_label))  # set grid label
            gl.ylocator = mticker.FixedLocator(
                np.arange(int(extent[2]) - 5 * res_grid, int(extent[3]) + 5 * res_grid, res_label))
            # ticks
            gl.xlabel_style = {'size': self.Fig.font_ticks["size"], 'family': self.Fig.font_ticks["family"]}
            gl.ylabel_style = {'size': self.Fig.font_ticks["size"], 'family': self.Fig.font_ticks["family"]}

        # title
        self.ax.set_title(title, fontdict=self.Fig.font_title)

    def axoff(self):
        ''' off the axis '''
        self.ax.axis("off")


def test():
    # general set
    root = "H"
    home = f"{root}:/research/flash_drough/"
    data_path = os.path.join(home, "GLDAS_Catchment/SoilMoist_RZ_tavg_19480101_20141230.npy")
    coord_path = os.path.join(home, "coord.txt")
    coord = pd.read_csv(coord_path, sep=",")
    sm_rz = np.load(data_path)
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
    # extend = [lon_min, lon_max, lat_min, lat_max]
    f = Figure()
    m = Map(f.ax, f, grid=True, res_grid=1, res_label=3)
    m.addmap(BaseMap())
    r = RasterMap(det, lat, lon, sm_rz_time_avg, expand=5)
    shape_file = [f"{root}:/GIS/Flash_drought/f'r_project.shp"]
    s = ShpMap(shape_file, facecolor="w", alpha=0.5)
    t = TextMap("Text", [lon[0], lat[0]], color="r")
    # img = ImgMap(img_file=['D:/NDVI/MODND1F.20000226.CN.NDVI.MAX.V2.TIF'], extent=[73.396, 134.8032, 3.7996, 53.6015])
    m.addmap(r)
    m.addmap(s)
    m.addmap(t)
    # m.addmap(img)
    # m.axoff()
    tif = TiffMap([f"F:/data/NDVI/MODND1F.20000306.CN.NDVI.MAX.V2.TIF"])
    m.addmap(tif)
    f.show()


if __name__ == "__main__":
    f = Figure()
    m = Map(f.ax, f, grid=True, res_grid=1, res_label=3)
    # m.addmap(BaseMap())
    tif = TiffMap([f"F:/data/NDVI/MODND1F.20000306.CN.NDVI.MAX.V2.TIF"])
    m.addmap(tif)
    f.show()

