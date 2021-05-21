# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# validation the sm data
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg' kg m-2 = mm ——> 10-3 m
# soil moisture data validation
# Soil moisture network: dm ——> 10-1m

import numpy as np
import pandas as pd
import os
import re
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import Workflow


class ReadFiletoExcel(Workflow.WorkBase):
    ''' Work, read International Soil Moisture Network(ISMN) txt file to excel, based on time index, time without data
     set to empty

     source data: https://ismn.geo.tuwien.ac.at/en/
    '''

    def run(self):
        ''' Implement WorkBase.run '''
        print("start read txt to excel")
        self.read_txt_to_excel()
        print("complete")

    def read_txt_to_excel(self):
        """ read txt file to excel
        txt files: each station have multiple sm layers(10)
        excel: m(time, some time do not have sm, set Nan) * n(layers)
        time without data set to empty
        """
        # general set
        network_china = 'H:/data_zxd/LP/SM_ISMN/CHINA'
        stations = os.listdir(network_china)

        # set date index
        years = list(range(1981, 2000))
        months = list(range(1, 13))
        days = [8, 18, 28]
        pd_index = [f"{year}/{month}/{day}" for year in years for month in months for day in days]  # time index
        pd_index = pd.to_datetime(pd_index)

        # read to excel
        for station in stations:
            # fill_value = np.NAN, then put txt sm(the time have sm) into Dataframe
            result = pd.DataFrame(np.full((len(pd_index), 11), fill_value=np.NAN), index=pd_index)
            stms = [os.path.join(network_china, station, d) for d in os.listdir(os.path.join(network_china, station)) if
                    d[-4:] == ".stm"]
            for i in range(len(stms)):
                with open(stms[i]) as f:
                    str_ = f.read()
                str_ = str_.splitlines()
                index_ = pd.to_datetime([i[:10] for i in str_[2:]])
                data_ = pd.Series([float(i[19:25]) for i in str_[2:]], index=index_)
                for j in range(len(data_)):
                    result.loc[data_.index[j], i] = data_.loc[data_.index[j]]
            result.to_excel(f"{station}.xlsx")

    def __repr__(self):
        return f"This is ReadFiletoExcel, read International Soil Moisture Network txt file to excel "

    def __str__(self):
        return f"This is ReadFiletoExcel, read International Soil Moisture Network txt file to excel"


class SmValidation(Workflow.WorkBase):
    ''' Work, validate Model Sm data(GLDAS Noah and GLDAS CLS) based on ISMN measured data(multiple stations) '''

    def __init__(self, model_sm, station_sm, model_coord, station_coord, model_date, station_name):
        ''' init function
        input:
            model_sm: 2D array, m(time) * n(grid), the model SM
            model_coord: pandas, the coord of Model SM corresponding to n(grid), read from coord.txt
            station_sm: list of pandas or pandas, the observation station SM, m * 1: index=time, col=sm
                    note: excel may have multiple sm, sum before put it in the class
            station_coord: list of pandas or pandas, the coord of all grid covering the station(in GIS), namely, average
                    Model sm between these grids to represent Model downscaling sm
            station_name: list, contains station names(one or more)

                    list of pandas: compare model_sm with multiple stations, and the sm-coord should be consistent
                    pandas: compare model_sm with one stations

            model_date: pandas.DatetimeIndex, the date covering model_sm and station_sm
                    use: pd.date_range('19480101', '20141230', freq='d') or pd.to_date()
        '''
        self.model_coord = model_coord
        self.station_coord = station_coord
        self.model_sm = model_sm
        self.station_sm = station_sm
        self.model_date = model_date
        self.station_name = station_name

    def run(self):
        ''' Implement WorkBase.run '''
        print("start validation")
        # compare model sm with one station
        if isinstance(self.station_coord, pd.DataFrame):
            model_downscaling_sm = self.downScaleModelSm(self.station_coord)
            station_sm = self.station_sm
            station_name = self.station_name[0]
            r, p_value = self.compare(station_sm, model_downscaling_sm, title=f"Compare datasets between model and"
                                                                              f" observation in {station_name}")
            ret = pd.DataFrame([r, p_value], index=['r', 'p_value'])

        # compare model sm with multiple stations
        elif isinstance(self.station_coord, list):
            ret = pd.DataFrame(np.zeros((2, len(self.station_name))), index=['r', 'p_value'], columns=self.station_name,
                               dtype='float')
            for i in range(len(self.station_coord)):
                model_downscaling_sm = self.downScaleModelSm(self.station_coord[i])
                station_sm = self.station_sm[i]
                station_name = self.station_name[i]
                r, p_value = self.compare(station_sm, model_downscaling_sm, title=f"Compare datasets between model and"
                                                                                  f" observation in {station_name}")
                ret.loc["r", self.station_name[i]] = r
                ret.loc["p_value", self.station_name[i]] = p_value

        print(ret)
        plt.show()
        print("complete")
        return ret

    def downScaleModelSm(self, station_coord):
        ''' downscale(spatial) Model sm, cal Model SM average into the coord of ISMN station
        Model sm(multiple time)
        grid: model_coord           Station: station_coord
        0     lon   lat                       lon    lat
        1     ...   ...        ->       0     ...   ...
        2     ...   ...
        3     ...   ...        ->       1     ...   ...
        ...   ...   ...        ->       ...   ...   ...
        n     ...   ...        ->       n'    ...   ...
        n' < n, each Station contains several grids in model_coord
        1) match and extrct: match -> based on station grids lon lat -> search in model_coord -> out: index, extract ->
           out: model_downscaling_sm(time, n' grids)
        2) average n' grids sm (model_downscaling_sm) -> out: model_downscaling_sm to represent Model sm downscaling on
           stations

        input:
            station_coord: the coord of grids covering the station
        out:
            model_downscaling_sm: 1D np.ndarray (time, ), Model downscaling sm on stations
        '''
        # general set
        model_sm = self.model_sm
        model_coord = self.model_coord

        # set double index "lon"/"lat" and insert a index col[0: len(model_coord)], namely, now can use lon/lat find
        # station coord index to match the model_coord
        model_coord = model_coord.set_index(["lon", "lat"])
        model_coord.insert(loc=0, column="index_", value=list(range(len(model_coord))))

        # cal downscaling model sm: average all grids Model sm in the station_coord
        # shape: m(time) * n(station corresponding coord)
        model_downscaling_sm = np.zeros((model_sm.shape[0], len(station_coord)), dtype="float")

        # match and extract station_coord grids Model sm based on matched coord(index) and put it into
        # model_downscaling_sm
        for i in range(len(station_coord)):
            # match station_coord and model_coord: to index
            index_ = model_coord.loc[(station_coord.loc[i, "lon"], station_coord.loc[i, "lat"])][0]
            model_downscaling_sm[:, i] = model_sm[:, int(index_)]

        # station grids average to represent Model downscaling sm on stations
        model_downscaling_sm = model_downscaling_sm.mean(axis=1)

        # unit change: value mm/1m -> value/1000 m/1m to match station unit
        model_downscaling_sm /= 1000

        return model_downscaling_sm

    def compare(self, station_sm, model_downscaling_sm, title="Compare datasets from model and observation"):
        ''' package the compare code: compare model sm with a station sm
        1) extract model data in the observation data date(exclude no data)
        2) plot compare time series fig
        3) plot bar fig
        4) calculate correlation coefficient
        input:
            station_sm: the observation data (dataframe) of the station, such as GUYUAN
            model_downscaling_sm: downscaling model sm on the station position

        output
            r, p_value: correlation analysis of model data and observation data
        '''
        # general set
        model_date = self.model_date
        station_sm.dropna(inplace=True)  # drop Nan
        station_date = station_sm.index

        # unit change
        station_sm = station_sm * 0.05

        # plot compare series
        self.plotCompareSeries(model_downscaling_sm, station_sm, station_date,  title=title)

        # extract data to the station date: model_date, m -> station date with values, m'
        model_downscaling_sm_extract_date = pd.DataFrame(model_downscaling_sm, index=model_date)
        model_downscaling_sm_extract_date = model_downscaling_sm_extract_date.loc[station_date].values.flatten()

        # plot compare bar
        self.plotCompareBar(station_date, model_downscaling_sm_extract_date, station_sm, title=title)

        # calculate the correlation between model data and station data
        r, p_value = pearsonr(model_downscaling_sm_extract_date, station_sm.values)

        return r, p_value

    def plotCompareSeries(self, model_downscaling_sm, station_sm, station_date,
                          title="Compare datasets between model and observation"):
        ''' plot time series to compare, compare model data with observation
        input
            model_downscaling_sm: downscaling model sm on the station position
            station_date: pandas.DatetimeIndex
            data: data correlate to date
        '''
        # general set
        model_date = self.model_date
        model_sm = model_downscaling_sm

        # plot compare time series
        plt.figure()
        plt.plot(model_date, model_sm, "royalblue", label="Model data", alpha=0.5)
        plt.plot(station_date, station_sm, "r-", label="Station data")
        plt.xlim(station_date.min(), station_date.max())
        font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
        font2 = {'family': 'Arial', 'weight': 'normal', 'size': 17}
        plt.xticks(fontproperties=font2)
        plt.yticks(fontproperties=font2)
        plt.xlabel("Date", font)
        plt.ylabel("Soil moisture / m", font)
        plt.title(title, font)
        plt.legend(prop=font2, loc='upper left', labelspacing=0.1, borderpad=0.2)

    def plotCompareBar(self, station_date, model_downscaling_sm, station_sm,
                       title="Compare datasets between model and observation"):
        ''' plot bar to compare, compare model data with observation '''
        # plot
        plt.figure()
        wid = 10
        plt.bar(station_date, model_downscaling_sm, color="royalblue", label="Model data", width=wid)
        plt.bar(station_date, -station_sm, color="r", label="Observation data", width=wid)
        plt.bar(station_date, model_downscaling_sm - station_sm, color="black", label="Diff", width=wid)
        # plt.bar(date_index, (data_model - data_observation)/data_model, color="green", label="Diff percentage",
        # width=wid)
        font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
        font2 = {'family': 'Arial', 'weight': 'normal', 'size': 17}
        plt.xticks(fontproperties=font2)
        h = plt.yticks(fontproperties=font2)
        plt.yticks(ticks=h[0], labels=['%2.2f' % abs(i) for i in h[0]], fontproperties=font2)
        plt.xlabel("Date", font)
        plt.ylabel("Soil moisture / m", font)
        plt.title(title, font)
        plt.legend(prop=font2, loc='upper left', labelspacing=0.1, borderpad=0.2)


def read_station_sm(start_layer, end_layer):
    ''' read station sm, start_layer and end_layer to set the sum layers
    0: 0-0.05
    1: 0.05-0.1
    2: 0.1-0.2
    3: 0.2-0.3
    4: 0.3-0.4
    5: 0.4-0.5
    6: 0.5-0.6
    7: 0.6-0.7
    8: 0.7-0.8
    9: 0.8-0.9
    10: 0.9-1.0

    e.g. 0-1 sum: start_layer=0, end_layer=10; 0.1-0.4 sum: 2-4
    '''
    station_coord = [os.path.join('G:/data_zxd/LP/SM_ISMN/CHINA/2.Station_coord', coord_) for coord_ in
                  os.listdir('G:/data_zxd/LP/SM_ISMN/CHINA/2.Station_coord')]
    station_sm = [os.path.join('G:/data_zxd/LP/SM_ISMN/CHINA/1.Station_sm', station_) for station_ in
                  os.listdir('G:/data_zxd/LP/SM_ISMN/CHINA/1.Station_sm')]

    # keep same order
    station_coord.sort()
    station_sm.sort()

    # extract station_name
    station_name = [station_sm_[station_sm_.rfind('\\') + 1: -5] for station_sm_ in station_sm]

    # read station_coord and station_sm
    station_coord = [pd.read_csv(station_coord_) for station_coord_ in station_coord]
    station_sm = [pd.read_excel(station_sm_, index_col=0) for station_sm_ in station_sm]

    # sum soil layers
    for i in range(len(station_sm)):
        station_sm[i]["sum"] = station_sm[i].iloc[:, start_layer: end_layer + 1].sum(axis=1, skipna=False)
        station_sm[i] = station_sm[i]["sum"]

    return station_sm, station_coord, station_name


def GLDAS_CLS_validation():
    # load data
    model_sm = np.load('H:/research/flash_drough/GLDAS_Catchment/SoilMoist_RZ_tavg_19480101_20141230.npy')
    model_coord = pd.read_csv("H:/GIS/Flash_drought/coord.txt")
    model_date = pd.to_datetime(model_sm[:, 0], format='%Y%m%d')  # %Y%m%d.%H%S

    # read station sm
    station_sm, station_coord, station_name = read_station_sm(0, 10)

    # validation
    smv = SmValidation(model_sm[:, 1:], station_sm, model_coord, station_coord, model_date, station_name)
    ret = smv.run()
    ret.to_excel("GLDAS_CLS_SoilMoist_RZ_tavg_Validation.xlsx")


def GLDAS_NOAH_RootMoist_validation():
    # load data
    model_sm = np.load('H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101_20141231_D.npy')
    model_coord = pd.read_csv("H:/GIS/Flash_drought/coord.txt")
    model_date = pd.to_datetime(model_sm[:, 0], format='%Y%m%d')

    # read station sm
    station_sm, station_coord, station_name = read_station_sm(0, 10)

    # validation
    smv = SmValidation(model_sm[:, 1:], station_sm, model_coord, station_coord, model_date, station_name)
    ret = smv.run()
    ret.to_excel("GLDAS_Noah_RootMoist_inst_Validation.xlsx")


def GLDAS_NOAH_SoilMoi0_10cm_inst_validation():
    # load data
    model_sm = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101_20141231_D.npy')
    model_coord = pd.read_csv("H:/GIS/Flash_drought/coord.txt")
    model_date = pd.to_datetime(model_sm[:, 0], format='%Y%m%d')

    # read station sm
    station_sm, station_coord, station_name = read_station_sm(0, 1)

    # validation
    smv = SmValidation(model_sm[:, 1:], station_sm, model_coord, station_coord, model_date, station_name)
    ret = smv.run()
    ret.to_excel("GLDAS_Noah_SoilMoi0_10cm_inst_Validation.xlsx")


def GLDAS_NOAH_SoilMoi0_100cm_inst_validation():
    # load data
    model_sm = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_D.npy')
    model_coord = pd.read_csv("H:/GIS/Flash_drought/coord.txt")
    model_date = pd.to_datetime(model_sm[:, 0], format='%Y%m%d')

    # read station sm
    station_sm, station_coord, station_name = read_station_sm(0, 10)

    # validation
    smv = SmValidation(model_sm[:, 1:], station_sm, model_coord, station_coord, model_date, station_name)
    ret = smv.run()
    ret.to_excel("GLDAS_Noah_SoilMoi0_100cm_inst_Validation.xlsx")


def GLDAS_NOAH_SoilMoi10_40cm_inst_validation():
    # load data
    model_sm = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101_20141231_D.npy')
    model_coord = pd.read_csv("H:/GIS/Flash_drought/coord.txt")
    model_date = pd.to_datetime(model_sm[:, 0], format='%Y%m%d')

    # read station sm
    station_sm, station_coord, station_name = read_station_sm(2, 4)

    # validation
    smv = SmValidation(model_sm[:, 1:], station_sm, model_coord, station_coord, model_date, station_name)
    ret = smv.run()
    ret.to_excel("GLDAS_Noah_SoilMoi10_40cm_inst_Validation.xlsx")


def GLDAS_NOAH_SoilMoi40_100cm_inst_validation():
    # load data
    model_sm = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101_20141231_D.npy')
    model_coord = pd.read_csv("H:/GIS/Flash_drought/coord.txt")
    model_date = pd.to_datetime(model_sm[:, 0], format='%Y%m%d')

    # read station sm
    station_sm, station_coord, station_name = read_station_sm(2, 4)

    # validation
    smv = SmValidation(model_sm[:, 1:], station_sm, model_coord, station_coord, model_date, station_name)
    ret = smv.run()
    ret.to_excel("GLDAS_Noah_SoilMoi40_100cm_inst_Validation.xlsx")


if __name__ == '__main__':
    # GLDAS_CLS_validation()
    # GLDAS_NOAH_RootMoist_validation()
    # GLDAS_NOAH_RootMoist_validation()
    # GLDAS_NOAH_SoilMoi0_100cm_inst_validation()
    # GLDAS_NOAH_SoilMoi10_40cm_inst_validation()
    # GLDAS_NOAH_SoilMoi40_100cm_inst_validation()
    pass
