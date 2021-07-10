# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# adjustment parameters in Drought and FD identification
# 1) params

import numpy as np
import pandas as pd
import FlashDrought
import Drought
import os
import draw_plot
import Workflow


class SensitivityDrought(Workflow.WorkBase):
    ''' Work, analyze the sensitivity of parameters to Drought Identification
    params:
        pc in pooling
        rds in excluding

    drought character to be compared:
        DD_mean, DS_mean, D_number

    ratio: drought character after pooling and excluding / drought character on baseline (without pooling and excluding)
    '''

    def __init__(self, drought_index, pc, rds, drought_threshold=0.4, tc=1, save_on=False, info=""):
        ''' init function
        input:
            drought_index: 1D list or numpy array, drought index
            drought_threshold: drought threshold, default=0.4
            tc: predefined critical duration in pooling process, i.e. 0, 0.1, ..., 0.5
            pc: pooling ratio, the critical ratio of excess volume(vi) of inter-event time and the preceding deficit
                    volume(si), which will be analyzing sensitivity to drought character, i.e. 0, 0.1, ..., 0.5
            rds: predefined critical excluding ratio, compare with rd/rs = d/s / d/s_mean,  which will be analyzing
                    sensitivity to drought character

            info: str, informatiom for this Class to print, shouldn't too long
            save_on: False or str, whether to save output into .xlsx, if do, input a save_path to save_on

        out:
            DD_ratio_pd, DS_ratio_pd, D_number_ratio: the ratio of character of drought after pooling to character of
                    drought on the baseline
        '''
        self._drought_index = drought_index
        self._drought_threshold = drought_threshold
        self._tc = tc
        self._pc = pc
        self._rds = rds
        self._save_on = save_on
        self._info = info

    def __call__(self):
        ''' Implement WorkBase.__call__ '''
        # baseline
        DD_mean_baseline, DS_mean_baseline, D_number_baseline = self.baselineCharacter()

        # pooling and excluding
        DD_mean, DS_mean, D_number = self.PoolingExcludingCharacter()

        # indicator = ratio of character to character_baseline
        DD_ratio = DD_mean / DD_mean_baseline
        DS_ratio = DS_mean / DS_mean_baseline
        D_number_ratio = D_number / D_number_baseline

        # plot
        self.plotSensitivity(DD_ratio=DD_ratio, DS_ratio=DS_ratio, D_number_ratio=D_number_ratio)

        # out
        index_ = ['pc=%.2f' % pc_ for pc_ in self._pc]
        columns_ = ['rds=%.2f' % rds_ for rds_ in self._rds]
        DD_ratio_pd = pd.DataFrame(DD_ratio, index=index_, columns=columns_)
        DS_ratio_pd = pd.DataFrame(DS_ratio, index=index_, columns=columns_)
        D_number_ratio_pd = pd.DataFrame(D_number_ratio, index=index_, columns=columns_)

        # save
        if self._save_on != False:
            DD_ratio_pd.to_excel(f"DD_ratio_pd_{self._save_on}.xlsx")
            DS_ratio_pd.to_excel(f"DS_ratio_pd_{self._save_on}.xlsx")
            D_number_ratio_pd.to_excel(f"D_number_ratio_pd_{self._save_on}.xlsx")

        return DD_ratio_pd, DS_ratio_pd, D_number_ratio_pd

    def baselineCharacter(self):
        ''' cal drought character(DD_mean, DS_mean, D_number) on the baseline '''
        drought_baseline = Drought.Drought(self._drought_index, Date_tick=[], threshold=0.4, pooling=False,
                                           excluding=False)
        out_drought_baseline = drought_baseline.out_put()
        DD_mean_baseline = out_drought_baseline["DD"].mean()
        DS_mean_baseline = out_drought_baseline["DS"].mean()
        D_number_baseline = len(out_drought_baseline)
        return DD_mean_baseline, DS_mean_baseline, D_number_baseline

    def PoolingExcludingCharacter(self):
        ''' cal drought character(DD_mean, DS_mean, D_number) after pooling and excluding with different params (pc, rds)
        '''
        pc = self._pc
        rds = self._rds

        # row: pc params, col: rds params
        DD_mean = np.zeros((len(pc), len(rds)))
        DS_mean = np.zeros((len(pc), len(rds)))
        D_number = np.zeros((len(pc), len(rds)))

        # cal characters based on different pooling and excluding params
        for i in range(len(pc)):
            for j in range(len(rds)):
                drought_ = Drought.Drought(self._drought_index, Date_tick=[], threshold=0.4, pooling=True, tc=self._tc,
                                           pc=pc[i], excluding=True, rds=rds[j])
                out_drought_ = drought_.out_put()
                DD_mean[i, j] = out_drought_["DD"].mean()
                DS_mean[i, j] = out_drought_["DS"].mean()
                D_number[i, j] = len(out_drought_)

        return DD_mean, DS_mean, D_number

    def plotSensitivity(self, DD_ratio, DS_ratio, D_number_ratio):
        ''' plot '''
        # general set
        f = draw_plot.FigureVert(3, hspace=0.3, sharex=True, sharey=True)
        DD_ratio = np.flip(DD_ratio, axis=0)  # flip to plot
        DS_ratio = np.flip(DS_ratio, axis=0)
        D_number_ratio = np.flip(D_number_ratio, axis=0)
        contournumber = 5
        linewidths = 0.5
        cmap = "hot_r"
        alpha = 0.5

        # plot DD ratio
        draw_DD = draw_plot.Draw(f.ax[0], f, title="Drought Duration Ratio", legend_on=False, labelx="rds", labely="pc")
        contourf_DD = draw_plot.ContourfDraw(self._rds, self._pc, DD_ratio, levels=contournumber, cb_label="ratio", cmap=cmap,
                                             alpha=alpha)
        contour_DD = draw_plot.ContourDraw(self._rds, self._pc, DD_ratio, levels=contournumber, colors='k',
                                           linewidths=linewidths)
        draw_DD.adddraw(contourf_DD)
        draw_DD.adddraw(contour_DD)

        # plot DS ratio
        draw_DS = draw_plot.Draw(f.ax[1], f, title="Drought Severity Ratio", legend_on=False, labelx="rds", labely="pc")
        contourf_DS = draw_plot.ContourfDraw(self._rds, self._pc, DS_ratio, levels=contournumber, cb_label="ratio", cmap=cmap,
                                             alpha=alpha)
        contour_DS = draw_plot.ContourDraw(self._rds, self._pc, DS_ratio, levels=contournumber, colors='k',
                                           linewidths=linewidths)
        draw_DS.adddraw(contourf_DS)
        draw_DS.adddraw(contour_DS)

        # plot D_number ratio
        draw_D_number = draw_plot.Draw(f.ax[2], f, title="Drought Number Ratio", legend_on=False, labelx="rds", labely="pc")
        contourf_D_number = draw_plot.ContourfDraw(self._rds, self._pc, D_number_ratio, levels=contournumber, cb_label="ratio",
                                                   cmap=cmap, alpha=alpha)
        contour_D_number = draw_plot.ContourDraw(self._rds, self._pc, D_number_ratio, levels=contournumber, colors='k',
                                                 linewidths=linewidths)
        draw_D_number.adddraw(contourf_D_number)
        draw_D_number.adddraw(contour_D_number)

        f.show()

        # save
        # if self._save_on != False:
        #     plt.savefig(f"fig/{self._save_on}.svg")

    def __repr__(self):
        return f"This is SensitivityDrought, info: {self._info}, analyze the sensitivity of parameters(pc, rds) to " \
               f"Drought Identification"

    def __str__(self):
        return f"This is SensitivityDrought, info: {self._info}, analyze the sensitivity of parameters(pc, rds) to " \
               f"Drought Identification"


class SensitivityFlashDrought(Workflow.WorkBase):
    ''' Work, analyze the sensitivity of parameters to Flash Drought Identification
    params:
        fd_pc in pooling
        fd_rds in excluding

    drought character to be compared:
        FDD_mean
        FDS_mean
        FD_number
    '''

    def __init__(self, drought_index, tc, pc, rds, fd_tc, fd_pc, drought_threshold=0.4, save_on=False, info=""):
        ''' init function
        input:
            drought_index: 1D list or numpy array, drought index
            Drought:
                tc, pc, rds: these are given params in FD sensitivity analysis
                drought_threshold: drought threshold, default=0.4
            Flash Drought:
                fd_tc, fd_pc: these will be performed sensitivity analysis

            info: str, informatiom for this Class to print, shouldn't too long
            save_on: False or str, whether to save output into .xlsx, if do, input a save_path to save_on

        out:

        '''
        self._drought_index = drought_index
        self._drought_threshold = drought_threshold
        self._tc = tc
        self._pc = pc
        self._rds = rds
        self._fd_tc = fd_tc
        self._fd_pc = fd_pc
        self._save_on = save_on
        self._info = info

    def __call__(self):
        ''' Implement WorkBase.__call__ '''
        # baseline
        FDD_mean_baseline, FDS_mean_baseline, FD_number_baseline = self.baselineCharacter()

        # pooling
        FDD_mean, FDS_mean, FD_number = self.PoolingCharacter()

        # indicator = ratio of character to character_baseline
        FDD_ratio = FDD_mean / FDD_mean_baseline
        FDS_ratio = FDS_mean / FDS_mean_baseline
        FD_number_ratio = FD_number / FD_number_baseline

        # plot
        self.plotSensitivity(FDD_ratio=FDD_ratio, FDS_ratio=FDS_ratio, FD_number_ratio=FD_number_ratio)

        # out
        index_ = ['fd_pc=%.2f' % fd_pc_ for fd_pc_ in self._fd_pc]
        FDD_ratio_pd = pd.DataFrame(FDD_ratio, index=index_)
        FDS_ratio_pd = pd.DataFrame(FDS_ratio, index=index_)
        FD_number_ratio_pd = pd.DataFrame(FD_number_ratio, index=index_)

        # save
        if self._save_on != False:
            FDD_ratio_pd.to_excel(f"FDD_ratio_pd_{self._save_on}.xlsx")
            FDS_ratio_pd.to_excel(f"FDS_ratio_pd_{self._save_on}.xlsx")
            FD_number_ratio_pd.to_excel(f"FD_number_ratio_pd_{self._save_on}.xlsx")

        return FDD_ratio_pd, FDS_ratio_pd, FD_number_ratio_pd

    def baselineCharacter(self):
        ''' cal fd character(FDD_mean, FDS_mean, FD_number) on the baseline '''
        fd_baseline = FlashDrought.FlashDrought(self._drought_index, Date_tick=[], threshold=self._drought_threshold,
                                                pooling=True, tc=self._tc, pc=self._pc, excluding=True, rds=self._rds,
                                                RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                                                fd_pooling=False)
        FDD_mean_baseline = fd_baseline.FDD_mean
        FDS_mean_baseline = fd_baseline.FDS_mean
        FD_number_baseline = sum(fd_baseline.dp)
        return FDD_mean_baseline, FDS_mean_baseline, FD_number_baseline

    def PoolingCharacter(self):
        ''' cal fd character(FDD_mean, FDS_mean, FD_number) after pooling (not excluding here) with different params (pc)
        '''
        fd_pc = self._fd_pc

        # len: pc params
        FDD_mean = np.zeros((len(fd_pc), ))
        FDS_mean = np.zeros((len(fd_pc), ))
        FD_number = np.zeros((len(fd_pc), ))

        # cal characters based on different pooling and excluding params
        for i in range(len(fd_pc)):
            fd_ = FlashDrought.FlashDrought(self._drought_index, Date_tick=[], threshold=self._drought_threshold,
                                            pooling=True, tc=self._tc, pc=self._pc, excluding=True, rds=self._rds,
                                            RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                                            fd_pooling=True, fd_tc=self._fd_tc, fd_pc=self._fd_pc[i])
            FDD_mean[i] = fd_.FDD_mean
            FDS_mean[i] = fd_.FDS_mean
            FD_number[i] = sum(fd_.dp)

        return FDD_mean, FDS_mean, FD_number

    def plotSensitivity(self, FDD_ratio, FDS_ratio, FD_number_ratio):
        ''' plot '''
        # general set
        f = draw_plot.FigureVert(3, hspace=0.3)  # , sharex=True, sharey=True
        FDD_ratio = np.flip(FDD_ratio, axis=0)  # flip to plot
        FDS_ratio = np.flip(FDS_ratio, axis=0)
        FD_number_ratio = np.flip(FD_number_ratio, axis=0)
        linewidths = 0.5
        color = "r"

        # plot FDD_ratio
        draw_FDD = draw_plot.Draw(f.ax[0], f, title="FD Duration Ratio", legend_on=False, labelx="fd_pc", labely="ratio")
        plot_FDD = draw_plot.PlotDraw(self._fd_pc, FDD_ratio, linewidth=linewidths, color=color)
        draw_FDD.adddraw(plot_FDD)

        # plot FDS_ratio
        draw_FDS = draw_plot.Draw(f.ax[1], f, title="FD Severity Ratio", legend_on=False, labelx="fd_pc", labely="ratio")
        plot_FDS = draw_plot.PlotDraw(self._fd_pc, FDS_ratio, linewidth=linewidths, color=color)
        draw_FDS.adddraw(plot_FDS)

        # plot FD_number_ratio
        draw_FD_number = draw_plot.Draw(f.ax[2], f, title="FD Number Ratio", legend_on=False, labelx="fd_pc", labely="ratio")
        plot_FD_number = draw_plot.PlotDraw(self._fd_pc, FD_number_ratio, linewidth=linewidths, color=color)
        draw_FD_number.adddraw(plot_FD_number)

        # f.show()

        # save
        # if self._save_on != False:
        #     plt.savefig(f"fig/{self._save_on}.svg")

    def __repr__(self):
        return f"This is SensitivityFlashDrought, info: {self._info}, analyze the sensitivity of parameters to Flash " \
               f"Drought Identification"

    def __str__(self):
        return f"This is SensitivityFlashDrought, info: {self._info}, analyze the sensitivity of parameters to Flash " \
               f"Drought Identification"


def droughtSensitivityAnalysis(Sm_percentile):
    pc = np.arange(0, 1.01, 0.01)
    rds = np.arange(0, 1.01, 0.01)
    for tc in range(1, 7):
        sd = SensitivityDrought(Sm_percentile, pc, rds, tc=tc, save_on=tc)  # f"{tc}"
        DD_ratio_pd, DS_ratio_pd, D_number_ratio = sd()


def fdSensitivityAnalysis(Sm_percentile):
    tc = 6
    pc = 0.5
    rds = 0.2
    fd_pc = np.arange(0, 1.01, 0.01)
    for fd_tc in range(1, 4):
        sfd = SensitivityFlashDrought(Sm_percentile, tc, pc, rds, fd_tc, fd_pc, drought_threshold=0.4, save_on=fd_tc, info="")
        FDD_ratio_pd, FDS_ratio_pd, FD_number_ratio_pd = sfd()


if __name__ == '__main__':
    home = 'H:/research/flash_drough/GLDAS_Noah'
    Sm_percentile = np.load(os.path.join(home, 'SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile_RegionMean.npy'))
    Sm_percentile = Sm_percentile[:, 1]

    # drought sensitivity analysis
    # droughtSensitivityAnalysis(Sm_percentile)

    # flash drought sensitivity analysis
    fdSensitivityAnalysis(Sm_percentile)
