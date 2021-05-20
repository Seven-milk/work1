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


class CombineNoahSm(Workflow.WorkBase):
    ''' Work, Combine SoilMoi0_10cm_inst/10_40/40_100 into SoilMoi0_100cm '''

    def __init__(self, home, save=False):
        self.Sm_path = [os.path.join(home, sm) for sm in
                        ['SoilMoi0_10cm_inst_19480101.0300_20141231.2100.npy',
                         'SoilMoi10_40cm_inst_19480101.0300_20141231.2100.npy',
                         'SoilMoi40_100cm_inst_19480101.0300_20141231.2100.npy']]
        self.save = save
        self._info = 'Noah Sm Combination 0-100cm'

    def run(self):
        print("start combine")
        print("load data")
        Sm0_10 = np.load(self.Sm_path[0], mmap_mode='r')
        Sm10_40 = np.load(self.Sm_path[1], mmap_mode='r')
        Sm40_100 = np.load(self.Sm_path[2], mmap_mode='r')
        print("load complete")
        print("sum")
        self.Sm0_100 = np.zeros_like(Sm0_10)
        self.Sm0_100[:, 0] = Sm0_10[:, 0]
        self.Sm0_100[:, 1:] = Sm0_10[:, 1:] + Sm10_40[:, 1:] + Sm40_100[:, 1:]
        print("sum complete")
        print("save")
        if self.save == True:
            np.save('SoilMoi0_100cm_inst_19480101.0300_20141231.2100.npy', self.Sm0_100)
        print("complete!")

    def __repr__(self):
        return f"This is CombineNoahSm, info: {self._info}, Combine SoilMoi0_10cm_inst/10_40/40_100 into SoilMoi0_100cm"

    def __str__(self):
        return f"This is CombineNoahSm, info: {self._info}, Combine SoilMoi0_10cm_inst/10_40/40_100 into SoilMoi0_100cm"


class UpscaleTime(Workflow.WorkBase):
    ''' Work, Upscale time series, such as, upscale 3H series to daily series (average or sum), upscale daily series to
        pentad series(average or sum)
    '''

    def __init__(self, original_series, multiple: int, up_method: callable = lambda x: sum(x)/len(x),
                 original_date=None, save_path=None, combine=True, info=""):
        ''' init function
        input:
            up_method: upscale method, default = lambda x: sum(x)/len(x) = mean, note 1 0 1 -> 2/3 is not 2/2
            original_series: 1D or 2D np.array, original series to upscale, when daily_series is a 2D array,
                            m(time) * n(other, such as grid points)
            original_date: 1D array like, original date corresponding to original series
            save_path: str, path to save, default=None, namely not save
            combine: whether combine upscale_date & upscale_series to output and save
            multiple: int, upscale time from original_series to objective_series
                D -> pentad: 5
                3H -> D: 8
                D -> Y: 365
            info: str, informatiom for this Class to print, shouldn't too long

        output:
            {self.save_path}_date.npy, {self.save_path}_series.npy: upscale date and series
            {self.save_path}.npy: combine output, the first col is upscale_date
        '''

        self.original_series = original_series
        self.up_method = up_method

        if isinstance(original_date, list) == True or isinstance(original_date, np.ndarray) == True:
            self.original_date = original_date
        else:
            self.original_date = np.arange(len(self.original_series))

        self.save_path = save_path
        self.multiple = multiple
        self._info = info
        self.combine = combine

    def run(self):
        ''' implement WorkBase.run '''
        print("start upScale")
        print("start calculate")
        upscale_date, upscale_series = self.upScale()
        print("complete calculate")

        # whether combine and save
        if self.combine == True:
            upscale_series = np.hstack((upscale_date.reshape(len(upscale_date), 1), upscale_series))
            if self.save_path != None:
                np.save(self.save_path, upscale_series)

            print("complete upScale")
            return upscale_series

        else:
            if self.save_path != None:
                np.save(self.save_path + '_date', upscale_date)
                np.save(self.save_path + '_series', upscale_series)

            print("complete upScale")
            return upscale_date, upscale_series

    def upScale(self):
        ''' up scale series '''

        # cal the series num which can be contain in cal period
        multiple = self.multiple
        up_method = self.up_method
        num_in = len(self.original_series) // multiple
        num_out = len(self.original_series) - num_in * multiple

        # del [-numout:] to make sure len(original_series) can be exact division by self.multiple
        if len(self.original_series.shape) > 1:
            original_series = self.original_series[:-num_out, :] if num_out != 0 else self.original_series
            upscale_series = np.zeros((num_in, self.original_series.shape[1]), dtype="float")
        else:
            original_series = self.original_series[:-num_out] if num_out != 0 else self.original_series
            upscale_series = np.zeros((num_in, ), dtype="float")

        original_date = self.original_date[:-num_out] if num_out != 0 else self.original_date
        upscale_date = np.zeros((num_in,), dtype="float")

        # cal upscale_series & upscale_date
        for i in range(num_in):
            if len(self.original_series.shape) > 1:
                upscale_series[i, :] = up_method(original_series[i * multiple: (i + 1) * multiple, :])  # axis=0
            else:
                upscale_series[i] = up_method(original_series[i * multiple: (i + 1) * multiple])

            # center date(depend on multiple, odd is center, even is the right of center)
            upscale_date[i] = original_date[i * multiple + multiple // 2]

        return upscale_date, upscale_series

    def __repr__(self):
        return f"This is UpscaleTime, info: {self._info}, Upscale time series"

    def __str__(self):
        return f"This is UpscaleTime, info: {self._info}, Upscale time series"


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
            for i in range(self._sm.shape[1]):
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


class CompareSmPercentile(Workflow.WorkBase):
    ''' Work, compare sm_rz_pentad and sm_percentile_rz_pentad: differences result from the fit and section
     calculation '''

    def __init__(self, sm_rz_pentad, sm_percentile_rz_pentad, info=""):
        ''' init function '''

        self.sm_rz_pentad = sm_rz_pentad
        self.sm_percentile_rz_pentad = sm_percentile_rz_pentad
        self._info = info

    def run(self):
        """ implement WorkBase.run """
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.sm_rz_pentad[:, 1], "b", alpha=0.5)
        ax2.plot(self.sm_percentile_rz_pentad[:, 1], "r.", markersize=1)

    def __repr__(self):
        return f"This is CompareSmPercentile, info: {self._info}, compare sm_rz_pentad and sm_percentile_rz_pentad"

    def __str__(self):
        return f"This is CompareSmPercentile, info: {self._info}, compare sm_rz_pentad and sm_percentile_rz_pentad"


def combine_Noah_SM():
    # combine Noah Sm, sum
    cns = CombineNoahSm(home='H:/research/flash_drough/GLDAS_Noah', save=True)
    cns.run()


def Upscale_Noah_D():
    # Upscale Noah from 3H to D
    original_series = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101.0300_20141231.2100.npy']
    save_path = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_D']

    for i in range(len(original_series)):
        original_series_ = np.load(original_series[i], mmap_mode='r')
        original_series_post = original_series_[:7, :]  # start from 0300, the first day contains 7 days rather than 8 days
        original_series_after = original_series_[7:, :]

        D_post = UpscaleTime(original_series=original_series_post[:, 1:], multiple=7,
                                original_date=original_series_post[:, 0], save_path=None,
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:]).run()
        D_after = UpscaleTime(original_series=original_series_after[:, 1:], multiple=8,
                                original_date=original_series_after[:, 0], save_path=None,
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:]).run()

        D_ = np.vstack((D_post, D_after))
        np.save(save_path[i], D_)


def Upscale_Noah_Pentad():
    # Upscale Noah from D to Pentad
    original_series = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_D.npy']
    save_path = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad']
    upscale_Noah = Workflow.WorkFlow()

    for i in range(len(original_series)):
        original_series_ = np.load(original_series[i], mmap_mode='r')
        upscale_Noah_ = UpscaleTime(original_series=original_series_[:, 1:], multiple=5,
                                original_date=original_series_[:, 0], save_path=save_path[i],
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:])
        upscale_Noah += upscale_Noah_

    upscale_Noah.runflow()


def Upscale_CLS_Pentad():
    original_series = 'H:/research/flash_drough/GLDAS_Catchment/SoilMoist_RZ_tavg_19480101_20141230.npy'
    save_path = 'H:/research/flash_drough/GLDAS_Catchment/SoilMoist_RZ_tavg_19480101_20141230_Pentad'
    original_series = np.load(original_series, mmap_mode='r')
    upscale_CLS = UpscaleTime(original_series=original_series[:, 1:], multiple=5, original_date=original_series[:, 0],
                              save_path=save_path, combine=True, info=save_path[save_path.rfind("/") + 1:])
    upscale_CLS.run()


if __name__ == '__main__':
    # # general set
    # home = "H:/research/flash_drough/"
    # data_path = os.path.join(home, "GLDAS_Catchment")
    # coord_path = os.path.join(home, "coord.txt")
    # coord = pd.read_csv(coord_path, sep=",")
    # date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
    # sm_rz = np.loadtxt(os.path.join(data_path, "SoilMoist_RZ_tavg.txt"), dtype="float", delimiter=" ")
    # sm_rz_Helong = np.loadtxt(os.path.join(home, "sm_rz_Helong.txt"), dtype="float", delimiter=" ")
    # sm_rz_noHelong = np.loadtxt(os.path.join(home, "sm_rz_noHelong.txt"), dtype="float", delimiter=" ")
    #
    # # CalPentad
    # sm_rz_CP = CalPentad(date, sm_rz, info="sm_rz_CP")
    # sm_rz_Helong_CP = CalPentad(date, sm_rz_Helong, info="sm_rz_Helong_CP")
    # sm_rz_noHelong_CP = CalPentad(date, sm_rz_noHelong, info="sm_rz_noHelong_CP")
    # WF_CP = Workflow.WorkFlow(sm_rz_CP, sm_rz_Helong_CP, sm_rz_noHelong_CP)
    #
    # date_pentad, sm_rz_pentad = sm_rz_CP.run()
    # _, sm_rz_Helong_pentad = sm_rz_Helong_CP.run()
    # _, sm_rz_noHelong_pentad = sm_rz_noHelong_CP.run()
    #
    # # CalSmPercentile
    # sm_rz_pentad_CSP = CalSmPercentile(sm_rz_pentad, timestep=73, info="sm_rz_pentad_CSP")
    #
    # sm_percentile_rz_pentad = sm_rz_pentad_CSP.run()
    #
    # # compare sm and sm_percentile
    # CompareSP = CompareSmPercentile(sm_rz_pentad, sm_percentile_rz_pentad, info="sm_rz_pentadVSsm_percentile_rz_pentad")
    #
    # # Pretreatment_data WF
    # PDWF = Workflow.WorkFlow()
    # PDWF += WF_CP
    # PDWF += sm_rz_pentad_CSP
    # PDWF += CompareSP
    # PDWF.runflow(key=[4])
