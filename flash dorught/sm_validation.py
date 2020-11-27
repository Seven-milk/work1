# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# validation the sm data
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg' kg m-2 = mm ——> 10-3 m
# Soil moisture network: dm ——> 10-1m

import numpy as np
import pandas as pd
import FDIP
import os
import re
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

# soil moisture data validation
home = "H:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment")
coord_path = "H:/GIS/Flash_drought/coord.txt"
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101', '20141230', freq='d')  # .strftime("%Y%m%d").to_numpy(dtype="int")
sm_rz = np.loadtxt(os.path.join(data_path, "SoilMoist_RZ_tavg.txt"), dtype="float", delimiter=" ")

# sm from NETWORK
sm_network_path = "H:/data_zxd/LP/SM_ISMN"
network_china = os.path.join(sm_network_path, "CHINA")

# china read txt to excel
stations = os.listdir(network_china)
years = list(range(1981, 2000))
months = list(range(1, 13))
days = [8, 18, 28]
pd_index = [f"{year}/{month}/{day}" for year in years for month in months for day in days]
pd_index = pd.to_datetime(pd_index)
# for station in stations:
#     result = pd.DataFrame(np.full((len(pd_index), 11), fill_value=np.NAN), index=pd_index)
#     stms = [os.path.join(network_china, station, d) for d in os.listdir(os.path.join(network_china, station)) if
#             d[-4:] == ".stm"]
#     for i in range(len(stms)):
#         with open(stms[i]) as f:
#             str_ = f.read()
#         str_ = str_.splitlines()
#         index_ = pd.to_datetime([i[:10] for i in str_[2:]])
#         data_ = pd.Series([float(i[19:25]) for i in str_[2:]], index=index_)
#         for j in range(len(data_)):
#             result.loc[data_.index[j], i] = data_.loc[data_.index[j]]
#     result.to_excel(f"{station}.xlsx")

names = locals()
stations = [os.path.join(network_china, d) for d in stations if d[-5:] == ".xlsx"]
for station in stations:
    Dataframe_ = pd.read_excel(station, index_col=0)
    nan_df = Dataframe_.isnull().any(axis=1)
    nan_index = nan_df[nan_df == True].index
    for i in range(len(nan_index)):
        Dataframe_.loc[nan_index[i], :] = np.NaN  # 空值处理：只要包含空值的行，都设置为空值
    Dataframe_ = Dataframe_.dropna(axis=0)  # 空值处理：删除na所在行
    Dataframe_.insert(loc=0, column="sum", value=Dataframe_.sum(axis=1) / 10)  # 求和+单位/100
    names[re.search(r"\\[A-Z]*\.xlsx", station)[0][1:-5]] = Dataframe_  # observation


# function to obtain model data with a given coord and calculate its average
def average_coord(source_data, source_coord, coord_):
    ''' function to obtain data with a given coord and calculate its average
    input:
        source_data
        source_coord: pandas, the source_data coord
        coord_: pandas, the coord to obtain data match with the source_coord
    output:
        average_data
    '''
    source_coord = source_coord.set_index(["lon", "lat"])
    source_coord.insert(loc=0, column="index_", value=list(range(len(source_coord))))
    average_data = np.full((source_data.shape[0], len(coord_)), fill_value=-9999, dtype="float")
    for i in range(len(coord_)):
        index_ = source_coord.loc[(coord_.loc[i, "lon"], coord_.loc[i, "lat"])][0]
        average_data[:, i] = source_data[:, int(index_)]
    average_data = average_data.mean(axis=1)
    average_data /= 1000
    return average_data


# plot function
def plot_compare(date_model, date_observation, data_model, data_observation, y=0, path="1"):
    ''' plot time series to compare, compare model data with observation
    input
        date: pandas.DatetimeIndex
        data: data correlate to date
    '''
    plt.figure()
    plt.plot(date_model, data_model, "royalblue", label="Model data", alpha=0.5)
    plt.plot(date_observation, data_observation, "r-", label="Observation data")
    plt.xlim(date_observation.min(), date_observation.max())
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
    plt.xticks(fontproperties=font2)
    plt.yticks(fontproperties=font2)
    plt.xlabel("Date", font)
    plt.ylabel("Root Zone Soil moisture / m", font)
    plt.title("Compare datasets from model and observation", font)
    plt.legend(prop=font2, loc='upper left', labelspacing=0.1, borderpad=0.2)
    plt.show()
    if y == 1:
        plt.savefig(path + '.tiff', dpi=350, bbox_inches='tight')


# bar function
def plot_bar(date_index, data_model, data_observation, y=0, path="1"):
    ''' plot hist to compare, compare model data with observation '''
    plt.figure()
    wid = 10
    plt.bar(date_index, data_model, color="royalblue", label="Model data", width=wid)
    plt.bar(date_index, -data_observation, color="r", label="Observation data", width=wid)
    plt.bar(date_index, data_model - data_observation, color="black", label="Diff", width=wid)
    # plt.bar(date_index, (data_model - data_observation)/data_model, color="green", label="Diff percentage", width=wid)
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}
    plt.xticks(fontproperties=font2)
    h = plt.yticks(fontproperties=font2)
    plt.yticks(ticks=h[0], labels=['%2.2f' % abs(i) for i in h[0]], fontproperties=font2)
    plt.xlabel("Date", font)
    plt.ylabel("Root Zone Soil moisture / m", font)
    plt.title("Compare datasets from model and observation", font)
    plt.legend(prop=font2, loc='upper left', labelspacing=0.1, borderpad=0.2)
    # plt.show()
    if y == 1:
        plt.savefig(path + '.tiff', dpi=350, bbox_inches='tight')


def compare(date, observation, model_data, path):
    ''' package the compare code:
    1) extract model data in the observation data date
    2) plot compare fig and save it
    input:
        date: the date of the model data, such as data
        observation: the observation data (dataframe) of the station, such as GUYUAN
        model_data: the model avg data, such as avg_guyuan
        path: the path to save fig, such as GUYUAN_compare

    output
    '''
    plot_compare(date_model=date, date_observation=observation.index, data_model=model_data,
                 data_observation=observation.iloc[:, 0], y=0, path=path)

    # extract data on the observation date and calculate the correlation between model data and observation data
    extract_date = observation.index
    avg_pd = pd.DataFrame(model_data, index=date)
    extract_model = avg_pd.loc[observation.index]
    extract_observation = observation.iloc[:, 0]
    plot_bar(date_index=observation.index, data_model=extract_model.values.flatten()
             , data_observation=observation.iloc[:, 0], y=0, path=path)
    r, p_value = pearsonr(extract_model.values.flatten(), observation.iloc[:, 0])
    return r, p_value


# GUTYUAN
coord_guyuan = pd.read_csv(os.path.join(network_china, "GUYUAN.txt"), sep=",")
avg_guyuan = average_coord(source_data=sm_rz, source_coord=coord, coord_=coord_guyuan)  # model

# HUANXIAN
coord_huanxian = pd.read_csv(os.path.join(network_china, "HUANXIAN.txt"), sep=",")
avg_huanxian = average_coord(source_data=sm_rz, source_coord=coord, coord_=coord_huanxian)

# TIANSHUI
coord_tianshui = pd.read_csv(os.path.join(network_china, "TIANSHUI.txt"), sep=",")
avg_tianshui = average_coord(source_data=sm_rz, source_coord=coord, coord_=coord_tianshui)

# TONGWEI
coord_tongwei = pd.read_csv(os.path.join(network_china, "TONGWEI.txt"), sep=",")
avg_tongwei = average_coord(source_data=sm_rz, source_coord=coord, coord_=coord_tongwei)

# XIFENGZH
coord_xifengzh = pd.read_csv(os.path.join(network_china, "XIFENGZH.txt"), sep=",")
avg_xifengzh = average_coord(source_data=sm_rz, source_coord=coord, coord_=coord_xifengzh)

# YONGNING
coord_yongning = pd.read_csv(os.path.join(network_china, "YONGNING.txt"), sep=",")
avg_yongning = average_coord(source_data=sm_rz, source_coord=coord, coord_=coord_yongning)

# compare
observations_ = ["GUYUAN", "HUANXIAN", "TIANSHUI", "TONGWEI", "XIFENGZH", "YONGNING"]
model_datas = [avg_guyuan, avg_huanxian, avg_tianshui, avg_tongwei, avg_xifengzh, avg_yongning]
paths_ = [i + "_compare_series" for i in observations_]
corr = pd.DataFrame(index=observations_, columns=["r", "p value"])
for i in range(len(observations_)):
    observation = names[observations_[i]]
    model_data_ = model_datas[i]
    path_ = paths_[i]
    r_, p_value_ = compare(date=date, observation=observation, model_data=model_data_, path=path_)
    corr.loc[observations_[i], "r"] = r_
    corr.loc[observations_[i], "p value"] = p_value_
