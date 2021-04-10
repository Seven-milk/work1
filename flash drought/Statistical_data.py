# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Statistical data: calculate statistical parameter of Drought and FD
import numpy as np
import pandas as pd
import FDIP
import os
import mannkendall_test

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
Drought_year_number = pd.read_excel(os.path.join(home, "Drought_year_number.xlsx"), index_col=0)
FD_year_number = pd.read_excel(os.path.join(home, "FD_year_number.xlsx"), index_col=0)
Num_point = 1166  # grid number
year = np.arange(1948, 2015)


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


# cal the month from a int date (19481019 -> 10)
def date2month(date: int) -> int:
    ''' calculate the month from a int date
    input
        date: int date, like 19481019
    return
        ret: int month, like 10 (1~12)
    '''
    ret = int((date % 10000 - date % 100) / 100)
    return ret


# cal the year from a int date (19481019 -> 1948)
def date2year(date: int) -> int:
    ''' calculate the year from a int date
    input
        date: int date, like 19481019
    return
        ret: int year, like 1948
    '''
    ret = int((date - date % 10000) / 10000)
    return ret


def cal_stat_params():
    # calculate the statistical params for all grid (one value for one grid) (define)
    Drought_FOC = np.full((Num_point,), np.NAN, dtype='float')  # Drought FOC
    Drought_number = np.full((Num_point,), np.NAN, dtype='float')  # Drought number
    DD_mean = np.full((Num_point,), np.NAN, dtype='float')  # DD timing mean (mean by drought events number)
    DS_mean = np.full((Num_point,), np.NAN, dtype='float')  # DS timing mean
    SM_min_mean = np.full((Num_point,), np.NAN, dtype='float')  # SM_min timing mean

    FD_FOC = np.full((Num_point,), np.NAN, dtype='float')  # FD FOC
    FD_number = np.full((Num_point,), np.NAN, dtype='float')  # FD number
    FDD_mean = np.full((Num_point,), np.NAN, dtype='float')  # FDD timing mean
    FDS_mean = np.full((Num_point,), np.NAN, dtype='float')  # FDS timing mean
    RI_mean_mean = np.full((Num_point,), np.NAN, dtype='float')  # RI_mean timing mean
    RI_max_mean = np.full((Num_point,), np.NAN, dtype='float')  # RI_max timing mean

    # calculate the statistical params for all grid (loop)
    for i in range(Num_point):
        # the sm_rz_pentad time series of every point
        FD_ = FDIP.FD(sm_rz_pentad[:, i], Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5,
                      pc=0.28,
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


def cal_season_params():
    # calculate season params in all grid (season feature) (one value for one grid) (define)
    # all define based on the start time
    Drought_spring = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in spring(3, 4, 5)
    Drought_summer = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in summer(6, 7, 8)
    Drought_autumn = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in autumn(9, 10, 11)
    Drought_winter = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in winter(12, 1, 2)

    FD_spring = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in spring
    FD_summer = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in summer
    FD_autumn = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in autumn
    FD_winter = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in winter

    # season flag, describe which season is most happen in this grid (spring: 1, summer: 2, autumn: 3, winter: 4)
    # (one value for one grid)
    Drought_season_Flag = np.full((Num_point,), np.NAN, dtype='int')  # Drought season Flag
    FD_season_Flag = np.full((Num_point,), np.NAN, dtype='int')  # FD season Flag

    # calculate season params for all grid (season feature) (loop)
    for i in range(Num_point):
        # the sm_rz_pentad time series of every point
        FD_ = FDIP.FD(sm_rz_pentad[:, i], Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5,
                      pc=0.28,
                      excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                      fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)

        # calculate season params for this grid
        Drought_spring_ = 0  # drought events number occur in spring(3, 4, 5)
        Drought_summer_ = 0  # drought events number occur in summer(6, 7, 8)
        Drought_autumn_ = 0  # drought events number occur in autumn(9, 10, 11)
        Drought_winter_ = 0  # drought events number occur in winter(12, 1, 2)
        FD_spring_ = 0  # FD events number occur in spring
        FD_summer_ = 0  # FD events number occur in summer
        FD_autumn_ = 0  # FD events number occur in autumn
        FD_winter_ = 0  # FD events number occur in winter

        for j in range(len(FD_.dry_flag_start)):
            Drought_month = date2month(date_pentad[FD_.dry_flag_start[j]])  # Drought start time
            if Drought_month in [3, 4, 5]:
                Drought_spring_ += 1
            elif Drought_month in [6, 7, 8]:
                Drought_summer_ += 1
            elif Drought_month in [9, 10, 11]:
                Drought_autumn_ += 1
            elif Drought_month in [12, 1, 2]:
                Drought_winter_ += 1
            for k in range(len(FD_.fd_flag_start[j])):
                FD_month = date2month(date_pentad[FD_.fd_flag_start[j][k]])  # FD start time
                if FD_month in [3, 4, 5]:
                    FD_spring_ += 1
                elif FD_month in [6, 7, 8]:
                    FD_summer_ += 1
                elif FD_month in [9, 10, 11]:
                    FD_autumn_ += 1
                elif FD_month in [12, 1, 2]:
                    FD_winter_ += 1

        Drought_spring[i] = Drought_spring_  # drought events number occur in spring(3, 4, 5)
        Drought_summer[i] = Drought_summer_  # drought events number occur in summer(6, 7, 8)
        Drought_autumn[i] = Drought_autumn_  # drought events number occur in autumn(9, 10, 11)
        Drought_winter[i] = Drought_winter_  # drought events number occur in winter(12, 1, 2)
        FD_spring[i] = FD_spring_  # FD events number occur in spring
        FD_summer[i] = FD_summer_  # FD events number occur in summer
        FD_autumn[i] = FD_autumn_  # FD events number occur in autumn
        FD_winter[i] = FD_winter_  # FD events number occur in winter

        # Drought season flag/ FD season flag: which season is most happen in this grid
        Drought_season_list = np.array([Drought_spring_, Drought_summer_, Drought_autumn_, Drought_winter_])
        FD_season_list = np.array([FD_spring_, FD_summer_, FD_autumn_, FD_winter_])
        Drought_season_Flag[i] = np.argmax(Drought_season_list) + 1
        FD_season_Flag[i] = np.argmax(FD_season_list) + 1

    # init dataframe to save season params
    season_params = pd.DataFrame(np.vstack((Drought_spring, Drought_summer, Drought_autumn, Drought_winter,
                                            FD_spring, FD_summer, FD_autumn, FD_winter,
                                            Drought_season_Flag, FD_season_Flag)).T,
                                 columns=("Drought_spring", "Drought_summer", "Drought_autumn", "Drought_winter",
                                          "FD_spring", "FD_summer", "FD_autumn", "FD_winter",
                                          "Drought_season_Flag", "FD_season_Flag"))

    # save to excel
    # season_params.to_excel("season_params.xlsx")


# special point: 1036, the grid with 376 drought events and 314 FD events
def SpecialPoint():
    FD_ = FDIP.FD(sm_rz_pentad[:, 1036], Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5,
                  pc=0.28,
                  excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                  fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
    FD_.general_out()


# year drought number and FD number
def year_number():
    Drought_year_number = np.zeros((Num_point, len(year)), dtype='int')  # drought events number yearly sum
    FD_year_number = np.zeros((Num_point, len(year)), dtype='int')  # FD events number yearly sum
    Drought_year_number = pd.DataFrame(Drought_year_number, index=list(range(Num_point)), columns=year)
    FD_year_number = pd.DataFrame(FD_year_number, index=list(range(Num_point)), columns=year)

    for i in range(Num_point):
        # the sm_rz_pentad time series of every point
        FD_ = FDIP.FD(sm_rz_pentad[:, i], Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5,
                      pc=0.28,
                      excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                      fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
        for j in range(len(FD_.dry_flag_start)):
            Drought_year = date2year(date_pentad[FD_.dry_flag_start[j]])
            Drought_year_number.loc[i, Drought_year] += 1
            for k in range(len(FD_.fd_flag_start[j])):
                FD_year = date2year(date_pentad[FD_.fd_flag_start[j][k]])
                FD_year_number.loc[i, FD_year] += 1

    # save to excel
    Drought_year_number.to_excel("Drought_year_number.xlsx")
    FD_year_number.to_excel("FD_year_number.xlsx")


# MK test for each grid: drought/FD year number
def mk_test_drought_FD_number():
    mk_drought = np.zeros((Num_point,), dtype=int)
    slope_drought = np.zeros((Num_point,), dtype=float)
    mk_FD = np.zeros((Num_point,), dtype=int)
    slope_FD = np.zeros((Num_point,), dtype=float)

    # drought year number
    for i in range(Num_point):
        Drought_number_ = Drought_year_number.iloc[i, :].values
        mk = mannkendall_test.MkTest(Drought_number_, x=year)
        if mk.mkret["trend"] == 1:
            mk_drought[i] = 1
            slope_drought[i] = mk.senret["slope"]
        elif mk.mkret["trend"] == -1:
            mk_drought[i] = -1
            slope_drought[i] = mk.senret["slope"]
        else:
            continue

    # FD year number
    for i in range(Num_point):
        FD_number_ = FD_year_number.iloc[i, :].values
        mk = mannkendall_test.MkTest(FD_number_, x=year)
        if mk.mkret["trend"] == 1:
            mk_FD[i] = 1
            slope_FD[i] = mk.senret["slope"]
        elif mk.mkret["trend"] == -1:
            mk_FD[i] = -1
            slope_FD[i] = mk.senret["slope"]
        else:
            continue

    # save to txt
    np.savetxt("mk_drought.txt", mk_drought)
    np.savetxt("mk_FD.txt", mk_FD)
    np.savetxt("slope_drought.txt", slope_drought)
    np.savetxt("slope_FD.txt", slope_FD)
