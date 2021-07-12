# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Typical events analysis: evolution over time
import Workflow
import os
import pandas as pd
import map_plot
import numpy as np


class TypicalEventAnalysis(Workflow.WorkBase):
    ''' Work, analyze typical event evaluation '''
    def __init__(self, info=""):
        self._info = info

    def __call__(self, *args, **kwargs):
        pass

    def dateIndex(self, date_data, date_start, date_end):
        ''' cal index of date_start(pd.Timestamp) and date_end(pd.Timestamp) in date_data(pd.DatetimeIndex) '''
        start_index = np.argwhere(date_data <= date_start).flatten()[-1]
        end_index = np.argwhere(date_data >= date_end).flatten()[0]
        return start_index, end_index

    def evaluationMultiVals(self, vals, date_datas, date_start, date_end, det, cb_label, extent, res_grid, res_label,
                            shps, cmap_name='BrBG'):
        ''' plot evaluation of Typical event for multiple vals
        input:
            vals: list of val, val m(time) * n(grid)
            date_datas: list of pd.DatetimeIndex(sorted), correspond to vals, len(date_datas) = len(vals),
                        len(date_data) = len(val)...
            date_start/end: pd.Timestamp
            cb_label: list of cb_label for each vals, if not plot cb, set None
            extent/res_grid/res_label: map plot set for the area
            shps: list of shp to plot
            cmap_name: str or list of str to set cmap
        '''
        start_index0, end_index0 = self.dateIndex(date_datas[0], date_start, date_end)
        time_number = end_index0 - start_index0 + 1
        vals_number = len(vals)
        fig = map_plot.Figure(figRow=time_number, figCol=vals_number, axflatten=False)

        for i in range(len(vals)):
            # set for vals[i]
            val = vals[i]
            date_data = date_datas[i]

            # extract period
            start_index, end_index = self.dateIndex(date_data, date_start, date_end)
            val = val[start_index: end_index + 1, :]
            date_data = date_data[start_index: end_index + 1]

            for j in range(val.shape[0]):
                if len(fig.ax.shape) > 1:
                    ax = fig.ax[j, i]
                else:
                    ax = fig.ax[j]
                cmap_name = cmap_name[i] if isinstance(cmap_name, list) else cmap_name
                map = map_plot.Map(ax, fig, extent=extent, grid=False, res_grid=res_grid, res_label=res_label,
                                   title=None, axoff=True)
                raster = map_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=val[j, :], expand=5,
                                            cmap_name=cmap_name, cb_label=cb_label[i])
                map.addmap(raster)
                for shp in shps:
                    map.addmap(shp)

        fig.show()

    def evaluationVal(self, val, date_data, date_start, date_end, det, cb_label, extent, res_grid, res_label, shps,
                   cmap_name='BrBG', axoff=True):
        ''' plot evaluation of Typical event for single val
        input:
            val: val m(time) * n(grid)
            date_data: pd.DatetimeIndex(sorted), correspond to val, len(date_data) = len(val)
            date_start/end: pd.Timestamp
            cb_label: str for setting cb label, if not plot cb, set None
            extent/res_grid/res_label: map plot set for the area
            shps: list of shp to plot
            cmap_name: str to set cmap
        '''
        start_index, end_index = self.dateIndex(date_data, date_start, date_end)
        time_number = end_index - start_index + 1
        fig = map_plot.Figure(time_number, wspace=0.01, hspace=0.01)
        val = val[start_index: end_index + 1, :]
        date_data = date_data[start_index: end_index + 1]

        for j in range(val.shape[0]):
            ax = fig.ax[j]
            map = map_plot.Map(ax, fig, extent=extent, grid=False, res_grid=res_grid, res_label=res_label,
                               title=None, axoff=axoff)
            raster = map_plot.RasterMap(det=det, data_lat=lat, data_lon=lon, data=val[j, :], expand=5,
                                        cmap_name=cmap_name, cb_label=cb_label)
            map.addmap(raster)
            for shp in shps:
                map.addmap(shp)

        fig.show()

    def evaluationValsegementMap(self, val, date_data, date_start, date_end, det, colorlevel, colordict, cbticks,
                                 cbticks_position, cb_label, extent, res_grid, res_label, shps, cmap_name='BrBG',
                                 axoff=True):
        ''' plot evaluation of Typical event for single val
        input:
            val: val m(time) * n(grid)
            date_data: pd.DatetimeIndex(sorted), correspond to val, len(date_data) = len(val)
            date_start/end: pd.Timestamp
            cb_label: str for setting cb label, if not plot cb, set None
            extent/res_grid/res_label: map plot set for the area
            shps: list of shp to plot
            cmap_name: str to set cmap
        '''
        start_index, end_index = self.dateIndex(date_data, date_start, date_end)
        time_number = end_index - start_index + 1
        fig = map_plot.Figure(time_number, wspace=0.01, hspace=0.01)
        val = val[start_index: end_index + 1, :]
        date_data = date_data[start_index: end_index + 1]

        for j in range(val.shape[0]):
            ax = fig.ax[j]
            map = map_plot.Map(ax, fig, extent=extent, grid=False, res_grid=res_grid, res_label=res_label,
                               title=None, axoff=axoff)
            raster = map_plot.RasterMap_segmented_cb(colorlevel=colorlevel, colordict=colordict, cbticks=cbticks,
                                                     cbticks_position=cbticks_position, det=det, data_lat=lat,
                                                     data_lon=lon, data=val[j, :], expand=5, cmap_name=cmap_name,
                                                     cb_label=cb_label)
            map.addmap(raster)
            for shp in shps:
                map.addmap(shp)

        fig.show()

    def __repr__(self):
        return f"This is TypicalEventAnalysis, analyze typical event evaluation, info: {self._info}"

    def __str__(self):
        return f"This is TypicalEventAnalysis, analyze typical event evaluation, info: {self._info}"


def evaluationMultiVals():
    tea.evaluationMultiVals(vals=[sm_pentad[:, 1:], sm_percentile_pentad[:, 1:]],
                            date_datas=[date_pentad, date_pentad],
                            date_start=date_start, date_end=date_end,
                            cb_label=[None, None], extent=extent, res_grid=res_grid,
                            res_label=res_label, shps=[boundry_shpMap], cmap_name='BrBG')


def evaluationVal():
    tea.evaluationVal(val=sm_percentile_pentad[:, 1:], date_data=date_pentad, date_start=date_start, date_end=date_end,
                      det=det, cb_label=None, extent=extent, res_grid=res_grid, res_label=res_label,
                      shps=[boundry_shpMap], cmap_name='BrBG')
    tea.evaluationVal(val=RI_grid_time_static.values, date_data=date_pentad, date_start=date_start,
                      date_end=date_end,
                      det=det, cb_label=None, extent=extent, res_grid=res_grid, res_label=res_label,
                      shps=[boundry_shpMap], cmap_name='BrBG')


def evaluationValsegementMap():
    val = drought_true_grid_time_static.values
    tea.evaluationValsegementMap(val=val, date_data=date_pentad, date_start=date_start, date_end=date_end, det=det,
                                 colorlevel=colorlevel, colordict=colordict, cbticks=cbticks,
                                 cbticks_position=cbticks_position, cb_label=None, extent=extent, res_grid=res_grid,
                                 res_label=res_label, shps=[boundry_shpMap], cmap_name='BrBG')
    val = fd_true_grid_time_static.values
    tea.evaluationValsegementMap(val=val, date_data=date_pentad, date_start=date_start, date_end=date_end, det=det,
                                 colorlevel=colorlevel, colordict=colordict, cbticks=cbticks,
                                 cbticks_position=cbticks_position, cb_label=None, extent=extent, res_grid=res_grid,
                                 res_label=res_label, shps=[boundry_shpMap], cmap_name='BrBG')


if __name__ == '__main__':
    # path
    root = "H"
    home = f"{root}:/research/flash_drough/"
    coord_path = os.path.join(home, "coord.txt")
    sm_pentad_path = os.path.join(home, "GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad.npy")
    sm_percentile_pentad_path = \
        os.path.join(home, "GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile.npy")
    sub_FlashDrought_path = "FlashDrought"
    drought_true_grid_time_static_path = os.path.join(home, "4.static_params", sub_FlashDrought_path,
                                                 "Drought_True_grid_time_static.csv")
    fd_true_grid_time_static_path = os.path.join(home, "4.static_params", sub_FlashDrought_path,
                                                 "FD_True_grid_time_static.csv")
    RI_grid_time_static_path = os.path.join(home, "4.static_params", sub_FlashDrought_path,
                                                 "RI_grid_time_static.csv")

    # read data
    coord = pd.read_csv(coord_path, sep=",")
    sm_pentad = np.load(sm_pentad_path)
    sm_percentile_pentad = np.load(sm_percentile_pentad_path)
    drought_true_grid_time_static = pd.read_csv(drought_true_grid_time_static_path, index_col=0)
    fd_true_grid_time_static = pd.read_csv(fd_true_grid_time_static_path, index_col=0)
    RI_grid_time_static = pd.read_csv(RI_grid_time_static_path, index_col=0)

    # date set
    date_pentad = pd.date_range('19480103', '20141231', freq='5d')
    date_d = pd.date_range('19480103', '20141231', freq='d')
    date_year = pd.date_range('19480103', '20141231', freq='Y')
    date_start = pd.Timestamp('19910101')
    date_end = pd.Timestamp('19920101')

    # map set
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
    boundry_shpMap = map_plot.ShpMap(boundry_shp, linewidth=0.1)

    # Drought / FD True or not
    colorlevel = [-0.5, 0.5, 1.5]
    colordict = ['w', 'saddlebrown']
    cbticks = ["NoDrought/FD", "Drought/FD"]
    cbticks_position = [0, 1]

    # typical event analysis
    tea = TypicalEventAnalysis()
    evaluationMultiVals()
    evaluationVal()
    evaluationValsegementMap()