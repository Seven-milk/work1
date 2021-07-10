# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Timing analysis: fixed spatial (one region), as a time series
# Root Zone Soil moisture: 'SoilMoist_RZ_tavg'
import numpy as np
import pandas as pd
import os
import draw_plot
import mannkendall_test
import variation_detect
from FlashDrought import FlashDrought_Frozen
import Workflow


class FDseries(Workflow.WorkBase):
    ''' Work, analyze FD series, namely, spatial or one point '''
    def __init__(self, drought_index, Date_tick, info=""):

        self.drought_index = drought_index
        self.Date_tick = Date_tick
        self._info = info

    def __call__(self):
        drought_index = self.drought_index
        Date_tick = self.Date_tick
        fd = FlashDrought_Frozen(drought_index, Date_tick)
        RI, out_put, dp = fd.general_out()
        return RI, out_put, dp

    def __repr__(self):
        return f"This is TimeSeriesAnalysis, analyze time series on spatial average or one point, info: {self._info}"

    def __str__(self):
        return f"This is TimeSeriesAnalysis, analyze time series on spatial average or one point, info: {self._info}"


class TimeSeriesAnalysis(Workflow.WorkBase):
    ''' Work, analyze time series, namely, it is spatial average or one point '''
    def __init__(self, x, vals, season=None, labelx="Date", labely="Y", info=""):
        ''' init function
        input:
            x: x axis for vals, len(x) shoule be equal to len(vals)
            vals: np.ndarray, the vals for further analysis
            season: default=None, if you wanna to analyze the season character of this vals, set it as a list like below
                    [spring, summer, autumn, winter]
            labelx: label for x
            labely: label for vals
            info: info for print

        output:
            plot for mktest / vdDetect(four detect methods) / seasonBoxplot
            ret = {"mkret": mkret, "senret": senret, "bgvd_bp": bgvd_bp, "sccvd_bp": sccvd_bp, "mkvd_bp": mkvd_bp, "ocvd_bp": ocvd_bp}
        '''
        self.x = x
        self.vals = vals
        self.labelx = labelx
        self.labely = labely
        self.season = season
        self._info = info

    def __call__(self):
        # general set
        x = self.x
        vals = self.vals
        labelx = self.labelx
        labely = self.labely
        season = self.season

        # time analyze
        mkret, senret = self.mkTest(x, vals, labelx=labelx, labely=labely)
        bgvd_bp, sccvd_bp, mkvd_bp, ocvd_bp = self.vdDetect(x, vals)
        if season != None:
            self.seasonBoxplot(*season, title="Boxplot", labely=labely)

        ret = {"mkret": mkret, "senret": senret, "bgvd_bp": bgvd_bp, "sccvd_bp": sccvd_bp, "mkvd_bp": mkvd_bp, "ocvd_bp": ocvd_bp}

        return ret

    def mkTest(self, x, vals, confidence: float = 0.95, labelx="X", labely="Y", title="MannKendall Test"):
        ''' mk test '''
        mk = mannkendall_test.MkTest(vals, confidence, x=x)
        mk.showRet(figure_on=True, num=1000, labelx=labelx, labely=labely, title=title)
        return mk.mkret, mk.senret

    def vdDetect(self, x, vals):
        ''' vd detect '''
        bgvd = variation_detect.BGVD(vals)
        sccvd = variation_detect.SCCVD(vals)
        mkvd = variation_detect.MKVD(vals)
        ocvd = variation_detect.OCVD(vals)
        bgvd.plot(time_ticks={"ticks": x, "interval": 10})
        sccvd.plot(time_ticks={"ticks": x, "interval": 10})
        mkvd.plot(time_ticks={"ticks": x, "interval": 10})
        ocvd.plot(time_ticks={"ticks": x, "interval": 10})
        return bgvd.bp, sccvd.bp, mkvd.bp, ocvd.bp

    def seasonBoxplot(self, spring, summer, autumn, winter, title="Boxplot", labelx=None, labely="y"):
        ''' season boxplot '''
        # season color
        facecolors = ['lightgreen', 'forestgreen', 'wheat', 'lightblue']
        fig_boxplot = draw_plot.Figure()
        draw_drought = draw_plot.Draw(fig_boxplot.ax, fig_boxplot, gridy=True, title=title, labelx=labelx, labely=labely)
        boxdraw = draw_plot.BoxDraw([spring, summer, autumn, winter], facecolors=facecolors,
                                    labels=["Spring", "Summer", "Autumn", "Winter"],
                                    notch=True, sym='r+', patch_artist=True, showfliers=False)
        draw_drought.adddraw(boxdraw)

    def __repr__(self):
        return f"This is TimeSeriesAnalysis, analyze time series on spatial average or one point, info: {self._info}"

    def __str__(self):
        return f"This is TimeSeriesAnalysis, analyze time series on spatial average or one point, info: {self._info}"


def DroughtYearNumberTimingAnalysis():
    tsa_drought = TimeSeriesAnalysis(x=date_year, vals=Drought_year_number.values.mean(axis=0), season=season_drought,
                                     labelx="Date", labely="Drought Number", info="drought number spatial average")
    ret = tsa_drought()
    return ret


def FDYearNumberTimingAnalysis():
    tsa_FD = TimeSeriesAnalysis(x=date_year, vals=FD_year_number.values.mean(axis=0), season=season_FD, labelx="Date",
                                labely="FD Number", info="FD number spatial average")
    ret = tsa_FD()
    return ret


def FDRegionMean():
    fds = FDseries(sm_percentile_region_mean[:, 1], date_pentad, info="SM percentile Region Mean")
    RI, out_put, dp = fds()
    return RI, out_put, dp


if __name__ == '__main__':
    # path
    root = "H"
    home = f"{root}:/research/flash_drough/"
    sm_percentile_region_mean_path =\
        os.path.join(home, "GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile_RegionMean.npy")
    drought_year_number_path = os.path.join(home, "4.static_params", "Drought_year_number.xlsx")
    FD_year_number_path = os.path.join(home, "4.static_params", "FD_year_number.xlsx")
    season_path = os.path.join(home, "4.static_params/season_static.xlsx")

    # read data
    sm_percentile_region_mean = np.load(sm_percentile_region_mean_path)
    Drought_year_number = pd.read_excel(drought_year_number_path, index_col=0)
    FD_year_number = pd.read_excel(FD_year_number_path, index_col=0)
    season_static = pd.read_excel(season_path, index_col=0)
    season_drought = [season_static["Drought_spring"].values, season_static["Drought_summer"].values,
                    season_static["Drought_autumn"].values, season_static["Drought_winter"].values]
    season_FD = [season_static["FD_spring"].values, season_static["FD_summer"].values,
                season_static["FD_autumn"].values, season_static["FD_winter"].values]

    # date set
    date_d = pd.date_range('19480101', '20141231', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")
    date_pentad = pd.date_range('19480103', '20141231', freq='5d').strftime("%Y").to_numpy(dtype="int")  # %m%d
    date_year = pd.date_range('19480101', '20141231', freq='Y').strftime("%Y").to_numpy(dtype="int")

    # timing analyze
    drought_year_number_ret = DroughtYearNumberTimingAnalysis()
    FD_year_number_ret = FDYearNumberTimingAnalysis()
    RI, out_put, dp = FDRegionMean()