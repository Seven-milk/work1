# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# pretreatment data and save
import numpy as np
import pandas as pd
import FDIP
import os
import re
from matplotlib import pyplot as plt

# general set
home = "H:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment")
coord_path = "H:/GIS/Flash_drought/coord.txt"
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
sm_rz = np.loadtxt(os.path.join(data_path, "SoilMoist_RZ_tavg.txt"), dtype="float", delimiter=" ")


# pentad/5days:  sm_rz/date to sm_rz_pentad/date_pentad
def cal_pentad(date, sm_rz):
    """ calculate sm_rz_pentad and date_pentad """
    num_pentad = len(date) // 5
    num_out = len(date) - num_pentad * 5
    sm_rz = sm_rz[:-1]
    date = date[:-1]
    date_pentad = np.full((num_pentad,), fill_value=-9999, dtype="int")
    sm_rz_pentad = np.full((num_pentad, len(coord)), fill_value=-9999, dtype="float")
    for i in range(num_pentad):
        sm_rz_pentad[i, :] = sm_rz[i * 5: (i + 1) * 5, :].mean(axis=0)
        date_pentad[i] = date[i * 5 + 2]
    return date_pentad, sm_rz_pentad


date_pentad, sm_rz_pentad = cal_pentad(date, sm_rz)
np.savetxt(os.path.join(home, "date_pentad.txt"), date_pentad)
np.savetxt(os.path.join(home, "sm_rz_pentad.txt"), sm_rz_pentad)


# pentad SM_moisture percentile: sm_rz_pentad to sm_percentile_rz_pentad
def cal_sm_percentile_rz_pentad(sm_rz_pentad):
    """ calculate sm_percentile_rz_pentad """
    sm_percentile_rz_pentad = np.full_like(sm_rz_pentad, fill_value=-9999, dtype="float")
    for i in range(sm_rz.shape[1]):
        SM_ = FDIP.SmPercentile(sm_rz_pentad[:, i], timestep=73)
        sm_percentile_rz_pentad[:, i] = SM_.SM_percentile
        print(i, "个计算完成")
    return sm_percentile_rz_pentad


sm_percentile_rz_pentad = cal_sm_percentile_rz_pentad(sm_rz_pentad)
np.savetxt(os.path.join(home, "sm_percentile_rz_pentad.txt"), sm_percentile_rz_pentad)


def compare_sm_sm_percentile():
    """ compare sm_rz_pentad and sm_percentile_rz_pentad: differences result from the fit and section calculation """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(sm_rz_pentad[:, 1], "b")
    ax2.plot(sm_percentile_rz_pentad[:, 1], "r.", markersize=1)
