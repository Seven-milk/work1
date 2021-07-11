# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Statistical data: calculate statistical parameter of Drought and FD
import numpy as np
import pandas as pd
import os
import mannkendall_test
import Workflow
# from FlashDrought import FlashDrought_Frozen as FD
from FlashDrought import FlashDrought_Liu_Frozen as FD
from useful_func import mean_list


class StaticalData(Workflow.WorkBase):
    ''' Work, cal statical data from original data '''

    def __init__(self, info=""):
        self._info = info

    def __call__(self):
        print("cal the sub function!")
        return None

    def gridDroughtFDStatistics(self, drought_index, Num_point, date_pentad, year=np.arange(1948, 2014), save_on=None):
        ''' calculate the drought and FD statistical params for each grid, namely, one value for one grid
        input:
            drought_index: drought index for FD, m(date) * n(grid)
            Num_point: grid number
            date_pentad: pd.DateTime, date_pentad in date_pentad[FD_.dry_flag_start[j]].month to count
                        Drought/FD_month/year for year number and season static
            year: year in MkTest(Drought_number_, x=year) and Drought_year_number = pd.DataFrame(Drought_year_number,
                index=list(range(Num_point)), columns=year), len(year)=length of the annual data, the range of year
                should be consistent with date_pentad
            save_on: None, not to save, or it can be save_path(str)

        output:
            ret: {"grid_static": grid_static, "season_static": season_static, "Drought_year_number": Drought_year_number,
               "FD_year_number": FD_year_number}, ret["mk_drought"], ret["slope_drought"], ret["mk_FD"], ret["slope_FD"]

                contains:
                    grid_static = ("Drought_FOC", "Drought_number", "DD_mean", "DS_mean", "index_min_mean", "FD_FOC", "FD_number",
                                 "FDD_mean", "FDS_mean", "RI_mean_mean", "RI_max_mean")
                    season_static = ("Drought_spring", "Drought_summer", "Drought_autumn", "Drought_winter", "FD_spring",
                                "FD_summer", "FD_autumn", "FD_winter", "Drought_season_Flag", "FD_season_Flag")
                    Drought_year_number
                    FD_year_number
        '''
        # general set
        Num_point = Num_point
        drought_index = drought_index
        Date_tick = []
        year = year
        date_pentad = date_pentad
        save_on = save_on

        # define variables
        # grid static
        Drought_FOC = np.full((Num_point,), np.NAN, dtype='float')  # Drought FOC
        Drought_number = np.full((Num_point,), np.NAN, dtype='float')  # Drought number
        DD_mean = np.full((Num_point,), np.NAN, dtype='float')  # DD timing mean (mean by drought events number)
        DS_mean = np.full((Num_point,), np.NAN, dtype='float')  # DS timing mean
        index_min_mean = np.full((Num_point,), np.NAN, dtype='float')  # SM_min timing mean

        FD_FOC = np.full((Num_point,), np.NAN, dtype='float')  # FD FOC
        FD_number = np.full((Num_point,), np.NAN, dtype='float')  # FD number
        FDD_mean = np.full((Num_point,), np.NAN, dtype='float')  # FDD timing mean
        FDS_mean = np.full((Num_point,), np.NAN, dtype='float')  # FDS timing mean
        RI_mean_mean = np.full((Num_point,), np.NAN, dtype='float')  # RI_mean timing mean
        RI_max_mean = np.full((Num_point,), np.NAN, dtype='float')  # RI_max timing mean

        # season static: all define based on the start time
        Drought_spring = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in spring(3, 4, 5)
        Drought_summer = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in summer(6, 7, 8)
        Drought_autumn = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in autumn(9, 10, 11)
        Drought_winter = np.full((Num_point,), np.NAN, dtype='int')  # drought events number occur in winter(12, 1, 2)

        FD_spring = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in spring
        FD_summer = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in summer
        FD_autumn = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in autumn
        FD_winter = np.full((Num_point,), np.NAN, dtype='int')  # FD events number occur in winter

        # season flag, describe which season is most happen in this grid (spring: 1, summer: 2, autumn: 3, winter: 4)
        Drought_season_Flag = np.full((Num_point,), np.NAN, dtype='int')  # Drought season Flag
        FD_season_Flag = np.full((Num_point,), np.NAN, dtype='int')  # FD season Flag

        # year number
        Drought_year_number = np.zeros((Num_point, len(year)), dtype='int')  # drought events number yearly sum
        FD_year_number = np.zeros((Num_point, len(year)), dtype='int')  # FD events number yearly sum
        Drought_year_number = pd.DataFrame(Drought_year_number, index=list(range(Num_point)), columns=year)
        FD_year_number = pd.DataFrame(FD_year_number, index=list(range(Num_point)), columns=year)

        # loop for calculating the statistical Drought and FD params of all grid
        print(f"there are {Num_point} grids")
        for i in range(Num_point):
            # FD for this grid
            FD_ = FD(drought_index[:, i], Date_tick)

            # cal grid static for this grid
            Drought_FOC[i] = FD_.DD.sum() / len(FD_.drought_index)  # FOC for Drought and FD
            FD_FOC[i] = sum([sum(x) for x in FD_.FDD]) / len(FD_.drought_index)

            Drought_number[i] = FD_.DD.shape[0]  # number for Drought and FD
            FD_number[i] = sum(FD_.dp)

            DD_mean[i] = FD_.DD.mean()  # DD_mean FDD_mean DS_mean FDS_mean
            FDD_mean[i] = mean_list([mean_list(x) for x in FD_.FDD])
            DS_mean[i] = FD_.DS.mean()
            FDS_mean[i] = mean_list([mean_list(x) for x in FD_.FDS])

            index_min_mean[i] = FD_.index_min.mean()  # index_min_mean

            RI_mean_mean[i] = mean_list([mean_list(x) for x in FD_.RImean])  # RI_mean_mean RI_max_mean
            RI_max_mean[i] = mean_list([mean_list(x) for x in FD_.RImax])

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
                Drought_month = date_pentad[FD_.dry_flag_start[j]].month
                Drought_year = date_pentad[FD_.dry_flag_start[j]].year

                Drought_year_number.loc[i, Drought_year] += 1
                for k in range(len(FD_.fd_flag_start[j])):
                    FD_year = date_pentad[FD_.fd_flag_start[j][k]].year
                    FD_year_number.loc[i, FD_year] += 1

                if Drought_month in [3, 4, 5]:
                    Drought_spring_ += 1
                elif Drought_month in [6, 7, 8]:
                    Drought_summer_ += 1
                elif Drought_month in [9, 10, 11]:
                    Drought_autumn_ += 1
                elif Drought_month in [12, 1, 2]:
                    Drought_winter_ += 1
                for k in range(len(FD_.fd_flag_start[j])):
                    FD_month = date_pentad[FD_.fd_flag_start[j][k]].month
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

            print(f"grid{i} complete")

        # init dataframe to save statistical params
        grid_static = pd.DataFrame(
            np.vstack((Drought_FOC, Drought_number, DD_mean, DS_mean, index_min_mean, FD_FOC, FD_number, FDD_mean,
                       FDS_mean, RI_mean_mean, RI_max_mean)).T,
            columns=("Drought_FOC", "Drought_number", "DD_mean", "DS_mean", "index_min_mean", "FD_FOC", "FD_number",
                     "FDD_mean", "FDS_mean", "RI_mean_mean", "RI_max_mean"))

        # init dataframe to save season params
        season_static = pd.DataFrame(
            np.vstack((Drought_spring, Drought_summer, Drought_autumn, Drought_winter, FD_spring, FD_summer, FD_autumn,
                       FD_winter, Drought_season_Flag, FD_season_Flag)).T,
            columns=("Drought_spring", "Drought_summer", "Drought_autumn", "Drought_winter", "FD_spring", "FD_summer",
                     "FD_autumn", "FD_winter", "Drought_season_Flag", "FD_season_Flag"))

        # save
        if save_on != None:
            grid_static.to_excel("grid_static.xlsx")
            season_static.to_excel("season_static.xlsx")
            Drought_year_number.to_excel("Drought_year_number.xlsx")
            FD_year_number.to_excel("FD_year_number.xlsx")

        # ret
        ret = {"grid_static": grid_static, "season_static": season_static, "Drought_year_number": Drought_year_number,
               "FD_year_number": FD_year_number}

        return ret

    def gridMkTest(self, x, vals, save_on=None):
        ''' calculate the mktest statistical params for each grid, namely, one value for one grid
        input:
            x: date for val, len(x) = len(val)
            vals: list for vals to do mktest, val: m(date) * n(grid)
        output:
            mk_ret, slope_ret: len = len(grid), for each grid, output its mktest ret
                                if there are several vals, mk_ret is a list for each mk_ret_
        '''
        # general set
        Num_point = vals[0].shape[1]

        # Drought_year_number FD_year_number
        mk_ret = []
        slope_ret = []

        # loop for calculating the mktest statistical params of all grid
        for val in vals:
            mk_ret_ = np.zeros((Num_point,), dtype=int)
            slope_ret_ = np.zeros((Num_point,), dtype=float)
            print(f"there are {Num_point} grids")
            for i in range(Num_point):
                val_ = val[:, i]
                mktest_ = mannkendall_test.MkTest(val_, x=x)
                if mktest_.mkret["trend"] == 1:
                    mk_ret_[i] = 1
                    slope_ret_[i] = mktest_.senret["slope"]
                elif mktest_.mkret["trend"] == -1:
                    mk_ret_[i] = -1
                    slope_ret_[i] = mktest_.senret["slope"]
                print(f"grid{i} complete")
            mk_ret.append(mk_ret_)
            slope_ret.append(slope_ret_)

        # save
        if save_on != None:
            np.save(save_on + "_mk_ret", mk_ret)
            np.save(save_on + "_slope_ret", slope_ret)

        return mk_ret, slope_ret

    def __repr__(self):
        return f"This is StaticalData, cal statical data from original data, info: {self._info}"

    def __str__(self):
        return f"This is StaticalData, cal statical data from original data, info: {self._info}"


if __name__ == '__main__':
    # path
    root = "H"
    home = f"{root}:/research/flash_drough/"
    sm_percentile_path =\
        os.path.join(home, "GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile.npy")
    Drought_year_number_path = os.path.join(home, "4.static_params", "FlashDrought_Liu", "Drought_year_number.xlsx")
    FD_year_number_path = os.path.join(home, "4.static_params", "FlashDrought_Liu", "FD_year_number.xlsx")
    year = np.arange(1948, 2015)

    # read data
    sm_percentile = np.load(sm_percentile_path)
    if os.path.exists(Drought_year_number_path):
        Drought_year_number = pd.read_excel(Drought_year_number_path, index_col=0)
    if os.path.exists(FD_year_number_path):
        FD_year_number = pd.read_excel(FD_year_number_path, index_col=0)

    # date set
    date_pentad = pd.to_datetime(sm_percentile[:, 0], format="%Y%m%d")
    sm_percentile = sm_percentile[:, 1:]

    # StaticalData
    std = StaticalData()
    grid_DFD_ret = std.gridDroughtFDStatistics(drought_index=sm_percentile, Num_point=1166, date_pentad=date_pentad,
                                               year=year, save_on="Drought_FD")

    # StaticalData for mktest of Drought_year_number and FD_year_number
    if os.path.exists(Drought_year_number_path):
        mk_ret_D_number, slope_ret_D_number = std.gridMkTest(year, [Drought_year_number.values.T], save_on="Drought_year_number")
    if os.path.exists(FD_year_number_path):
        mk_ret_FD_number, slope_ret_FD_number = std.gridMkTest(year, [FD_year_number.values.T], save_on="FD_year_number")