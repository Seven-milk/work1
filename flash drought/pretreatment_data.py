# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# pretreatment data and save
import numpy as np
import pandas as pd
import FDIP
import os
from matplotlib import pyplot as plt
import Workflow

# general set
home = "H:/research/flash_drough/"
data_path = os.path.join(home, "GLDAS_Catchment")
coord_path = os.path.join(home, "coord.txt")
coord = pd.read_csv(coord_path, sep=",")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
sm_rz = np.loadtxt(os.path.join(data_path, "SoilMoist_RZ_tavg.txt"), dtype="float", delimiter=" ")
sm_rz_Helong = np.loadtxt(os.path.join(home, "sm_rz_Helong.txt"), dtype="float", delimiter=" ")
sm_rz_noHelong = np.loadtxt(os.path.join(home, "sm_rz_noHelong.txt"), dtype="float", delimiter=" ")


class CalPentad(Workflow.WorkBase):
    ''' Work, calculate pentad/5days series from daily series, i.e. sm_rz/date to sm_rz_pentad/date_pentad '''

    def __init__(self, date, daily_series, save_path=None, info=""):
        ''' init function
        input:
            date: 1D array like, daily date, corresponding to daily_series
            daily_series: 1D or 2D np.array, when daily_series is a 2D array, m(time) * n(other, such as grid points)
            save_path: str, home path to save, if save_path=None(default), do not save
            info: str, informatiom for this Class to print and save in save_path, shouldn't too long

        output:
            pentad_date, pentad_series: pentad series from date and daily_series
        '''
        self.date = date
        self.daily_series = daily_series
        self.save_path = save_path
        self._info = info

    def run(self):
        ''' implement WorkBase.run '''
        num_pentad = len(self.date) // 5
        num_out = len(self.date) - num_pentad * 5

        # del [-numout:] to make sure len(daily_series) can be exact division by 5/pentad
        if len(self.daily_series.shape) > 1:
            daily_series = self.daily_series[:-num_out, :]
            pentad_series = np.full((num_pentad, daily_series.shape[1]), fill_value=-9999, dtype="float")
        else:
            daily_series = self.daily_series[:-num_out]
            pentad_series = np.full((num_pentad,), fill_value=-9999, dtype="float")

        date = self.date[:-num_out]
        pentad_date = np.full((num_pentad,), fill_value=-9999, dtype="int")

        # cal pentad_date & pentad_series
        for i in range(num_pentad):
            if len(self.daily_series.shape) > 1:
                pentad_series[i, :] = daily_series[i * 5: (i + 1) * 5, :].mean(axis=0)
            else:
                pentad_series[i, :] = daily_series[i * 5: (i + 1) * 5].mean(axis=0)
            pentad_date[i] = date[i * 5 + 2]  # center date

        # save result
        if self.save_path != None:
            np.savetxt(os.path.join(self.save_path, f"pentad_date_{self._info}.txt"), pentad_date)
            np.savetxt(os.path.join(self.save_path, f"pentad_series_{self._info}.txt"), pentad_series)

        return pentad_date, pentad_series

    def __repr__(self):
        return f"This is CalPentad, info: {self._info}, calculate pentad/5days series from daily series"

    def __str__(self):
        return f"This is CalPentad, info: {self._info}, calculate pentad/5days series from daily series"


class CalSmPercentile(Workflow.WorkBase):
    ''' Work, calculate SmPercentile series from SM series, i.e. sm_rz_pentad to sm_percentile_rz_pentad '''
    def __init__(self, sm, timestep, save_path=None, info=""):
        ''' init function
        input:
            sm: 1D or 2D array like, soil moisture series, m(time) * n(other, such as grid points)
            timestep: timestep in FDIP.SmPercentile
                365 : daily, 365 data in one year
                12 : monthly, 12 data in one year
                73 : pentad(5), 73 data in one year
                x ：x data in one year
            save_path: str, home path to save, if save_path=None(default), do not save
            info: str, informatiom for this Class to print and save in save_path, shouldn't too long

        output:

        '''
        self._sm = sm
        self.timestep = timestep
        self.save_path = save_path
        self._info = info

    def run(self):
        ''' implement WorkBase.run '''
        sm_percentile = np.full_like(self._sm, fill_value=-9999, dtype="float")

        if len(self._sm.shape) > 1:
            print(f"all series number {self._sm.shape[1]}")
            for i in range(sm_rz.shape[1]):
                SM_ = FDIP.SmPercentile(self._sm[:, i], timestep=self.timestep)
                sm_percentile[:, i] = SM_.SM_percentile
                print(f"sm series {i} calculated completely")
        else:
            SM_ = FDIP.SmPercentile(self._sm, timestep=self.timestep)
            sm_percentile = SM_.SM_percentile

        # save result
        if self.save_path != None:
            np.savetxt(os.path.join(self.save_path, f"sm_percentile_{self._info}.txt"), sm_percentile)

        return sm_percentile

    def __repr__(self):
        return f"This is CalSmPercentile, info: {self._info}, calculate SmPercentile series from SM series"

    def __str__(self):
        return f"This is CalSmPercentile, info: {self._info}, calculate SmPercentile series from SM series"


def compare_sm_sm_percentile():
    """ compare sm_rz_pentad and sm_percentile_rz_pentad: differences result from the fit and section calculation """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(sm_rz_pentad[:, 1], "b", alpha=0.5)
    ax2.plot(sm_percentile_rz_pentad[:, 1], "r.", markersize=1)


if __name__ == '__main__':
    # CalPentad
    sm_rz_CP = CalPentad(date, sm_rz, info="sm_rz_CP")
    sm_rz_Helong_CP = CalPentad(date, sm_rz_Helong, info="sm_rz_Helong_CP")
    sm_rz_noHelong_CP = CalPentad(date, sm_rz_noHelong, info="sm_rz_noHelong_CP")
    WF_CP = Workflow.WorkFlow(sm_rz_CP, sm_rz_Helong_CP, sm_rz_noHelong_CP)

    date_pentad, sm_rz_pentad = sm_rz_CP.run()
    _, sm_rz_Helong_pentad = sm_rz_Helong_CP.run()
    _, sm_rz_noHelong_pentad = sm_rz_noHelong_CP.run()

    # CalSmPercentile
    sm_rz_pentad_CSP = CalSmPercentile(sm_rz_pentad, timestep=73, info="sm_rz_pentad")

    sm_percentile_rz_pentad = sm_rz_pentad_CSP.run()

    # Pretreatment_data WF
    PDWF = Workflow.WorkFlow()
    PDWF += WF_CP
    PDWF += sm_rz_pentad_CSP

    # compare sm and sm_percentile
    compare_sm_sm_percentile()
