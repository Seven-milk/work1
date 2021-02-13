# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Statistical data: calculate statistical parameter of Drought and FD
import numpy as np
import pandas as pd
import FDIP
import os

# general set
home = "F:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment/SoilMoist_RZ_tavg.txt")
coord_path = os.path.join(home, "coord.txt")
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
date_pentad = np.loadtxt(os.path.join(home, "date_pentad.txt"), dtype="int")
sm_rz_pentad = np.loadtxt(os.path.join(home, "sm_rz_pentad.txt"))
sm_percentile_rz_pentad = np.loadtxt(os.path.join(home, "sm_percentile_rz_pentad.txt"), dtype="float", delimiter=" ")
Num_point = 1166  # grid number

# calculate the statistical parameters of each grid (one value for one grid)
Drought_FOC = np.full((Num_point, ), np.NAN, dtype='float')  # Drought FOC
Drought_number = np.full((Num_point, ), np.NAN, dtype='float')  # Drought number
DD_mean = np.full((Num_point, ), np.NAN, dtype='float')  # DD timing mean (mean by drought events number)
DS_mean = np.full((Num_point, ), np.NAN, dtype='float')  # DS timing mean
SM_min_mean = np.full((Num_point, ), np.NAN, dtype='float')  # SM_min timing mean

FD_FOC = np.full((Num_point, ), np.NAN, dtype='float')  # FD FOC
FD_number = np.full((Num_point, ), np.NAN, dtype='float')  # FD number
FDD_mean = np.full((Num_point, ), np.NAN, dtype='float')  # FDD timing mean
FDS_mean = np.full((Num_point, ), np.NAN, dtype='float')  # FDS timing mean
RI_mean_mean = np.full((Num_point, ), np.NAN, dtype='float')  # RI_mean timing mean
RI_max_mean = np.full((Num_point, ), np.NAN, dtype='float')  # RI_max timing mean


# cal the mean of a list
def mean_list(lst: list):
    ''' calculate the mean of a list
    input
        lst: a list have int or float.. number
    return
        ret: the mean of the list input
    '''
    if len(lst) != 0:
        ret = sum(lst) / len(lst)
    else:
        ret = 0
    return ret


# calculate the statistical params for all grid
for i in range(Num_point):
    # the sm_rz_pentad time series of every point
    FD_ = FDIP.FD(sm_rz_pentad[:, i], Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5, pc=0.28,
                  excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                  fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
    # SM_percentile, RI, out_put, dp = FD_.general_out()
    # Drought_FOC FD_FOC
    Drought_FOC[i] = FD_.DD.sum() / len(FD_.SM)
    FD_FOC[i] = sum([sum(x) for x in FD_.FDD]) / len(FD_.SM)
    # Drought_number FD_number
    Drought_number[i] = FD_.DD.shape[0]
    FD_number[i] = sum(FD_.dp)
    # DD_mean FDD_mean DS_mean FDS_mean
    DD_mean[i] = FD_.DD.mean()
    FDD_mean[i] = mean_list([mean_list(x) for x in FD_.FDD])
    DS_mean[i] = FD_.DS.mean()
    FDS_mean[i] = mean_list([mean_list(x) for x in FD_.FDS])
    # SM_min_mean
    SM_min_mean[i] = FD_.SM_min.mean()
    # RI_mean_mean RI_max_mean
    RI_mean_mean[i] = mean_list([mean_list(x) for x in FD_.RImean])
    RI_max_mean[i] = mean_list([mean_list(x) for x in FD_.RImax])

# init dataframe to save statistical params
static_params = pd.DataFrame(
    np.vstack((Drought_FOC, Drought_number, DD_mean, DS_mean, SM_min_mean, FD_FOC, FD_number, FDD_mean,
               FDS_mean, RI_mean_mean, RI_max_mean)).T,
    columns=("Drought_FOC", "Drought_number", "DD_mean", "DS_mean", "SM_min_mean", "FD_FOC", "FD_number",
             "FDD_mean", "FDS_mean", "RI_mean_mean", "RI_max_mean"))

# save to excel
static_params.to_excel("static_params.xlsx")
