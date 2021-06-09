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
from matplotlib import pyplot as plt
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

        out:
            DD_ratio_pd, DS_ratio_pd, D_number_ratio: the ratio of character of drought after pooling to character of
                    drought on the baseline
        '''
        self._drought_index = drought_index
        self._drought_threshold = drought_threshold
        self._tc = tc
        self._pc = pc
        self._rds = rds
        self.save_on = save_on
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
        D_number_ratio = pd.DataFrame(D_number_ratio, index=index_, columns=columns_)

        # save
        if self.save_on == True:
            DD_ratio_pd.to_excel("DD_ratio_pd.xlsx")
            DS_ratio_pd.to_excel("DS_ratio_pd.xlsx")
            D_number_ratio.to_excel("D_number_ratio.xlsx")

        return DD_ratio_pd, DS_ratio_pd, D_number_ratio

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
        contournumber = 10
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

    def __repr__(self):
        return f"This is SensitivityDrought, info: {self._info}, analyze the sensitivity of parameters(pc, rds) to " \
               f"Drought Identification"

    def __str__(self):
        return f"This is SensitivityDrought, info: {self._info}, analyze the sensitivity of parameters(pc, rds) to " \
               f"Drought Identification"


if __name__ == '__main__':
    home = 'H:/research/flash_drough/GLDAS_Noah'
    Sm_percentile = np.load(os.path.join(home, 'SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile_RegionMean.npy'))
    Sm_percentile = Sm_percentile[:, 1]
    pc = np.arange(0, 0.51, 0.01)
    rds = np.arange(0, 0.51, 0.01)
    sd = SensitivityDrought(Sm_percentile, pc, rds, tc=6, save_on=True)
    DD_ratio_pd, DS_ratio_pd, D_number_ratio = sd()










#
#
# # sensitity analysis: drought
# def plot_subplot(x: list, x_label: str, ratio_DD_mean: list, ratio_DS_mean: list, ratio_number: list,
#                  ratio_SM_min_mean: list):
#     """ plot drought properties changing with pc or rds(x)"""
#     font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
#     font_ticks = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
#     font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
#     plt.figure()
#     plt.subplot(2, 2, 1)
#     plt.plot(x, ratio_DD_mean, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(1, )
#     plt.title("Duration", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(2, 2, 2)
#     plt.plot(x, ratio_DS_mean, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(1, )
#     plt.title("Severity", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(2, 2, 3)
#     plt.plot(x, ratio_number, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(top=1)
#     plt.title("Events number", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(2, 2, 4)
#     plt.plot(x, ratio_SM_min_mean, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(top=1.02)
#     plt.title("Peak Value", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplots_adjust(hspace=0.35, wspace=0.2)
#     plt.show()
#
#
# # sensitity analysis: drought
# def fd_plot_subplot(x: list, x_label: str, ratio_FDD_mean: list, ratio_FDS_mean: list, ratio_number: list,
#                     ratio_number_NFD: list, ratio_RImean_mean: list, ratio_RImax_mean: list):
#     """ plot drought properties changing with pc or rds(x)"""
#     font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
#     font_ticks = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
#     font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
#     plt.figure()
#     plt.subplot(3, 2, 1)
#     plt.plot(x, ratio_FDD_mean, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(1, )
#     plt.title("FD Duration", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(3, 2, 2)
#     plt.plot(x, ratio_FDS_mean, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(1, )
#     plt.title("FD Severity", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(3, 2, 3)
#     plt.plot(x, ratio_number, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(top=1)
#     plt.title("FD Events number", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(3, 2, 4)
#     plt.plot(x, ratio_number_NFD, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(0.5, 1.5)
#     plt.title("NFD Events number", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(3, 2, 5)
#     plt.plot(x, ratio_RImean_mean, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(top=1)
#     plt.title("FD RI Mean", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplot(3, 2, 6)
#     plt.plot(x, ratio_RImax_mean, linestyle="-", marker="o", markersize=3)
#     plt.xlim(0, 0.5)
#     # plt.ylim(1, )
#     plt.title("FD RI Max", font_title)
#     plt.ylabel("Ratio of Mean", font_label)
#     plt.xlabel(x_label, font_label)
#     plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
#     plt.yticks(fontproperties=font_ticks)
#
#     plt.subplots_adjust(hspace=0.55, wspace=0.2)
#     plt.show()
#
#
# def sensitity_FD():
#     # sensitity analysis: under different pooling/excluding parameters (fd_tc, fd_pc, fd_rds)
#     fd_tc_avg = 2
#     pc_avg = 0.28
#     rds_avg = 0.22  # has been pooling and excluding in drought perspective
#     FD_avg = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
#                              pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
#                              eliminate_threshold=0.2,
#                              fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
#
#     # sensitity analysis: (pooling at fd_pc = 0 : 0.1 : 0.5 & no excluding) vs (no pooling & no excluding)
#     fd_pc_avg = [pc_ / 100 for pc_ in list(range(0, 51, 1))]
#     ratio_FDD_mean_p, ratio_FDS_mean_p, ratio_number_p, ratio_number_NFD_p, ratio_RImean_mean_p, ratio_RImax_mean_p = [], [], [], [], [], []
#     for i in range(len(fd_pc_avg)):
#         FD_ = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
#                               pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
#                               eliminate_threshold=0.2,
#                               fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_avg[i], fd_excluding=False, fd_rds=0.41)
#         ratio_FDD_mean_p.append(FD_.FDD_mean / FD_avg.FDD_mean)
#         ratio_FDS_mean_p.append(FD_.FDS_mean / FD_avg.FDS_mean)
#         ratio_number_p.append(sum(FD_.dp) / sum(FD_avg.dp))
#         ratio_number_NFD_p.append(FD_.dp.count(0) / FD_avg.dp.count(0))  # Number of Drought without flash drought
#         ratio_RImean_mean_p.append(
#             np.array([i for j in FD_.RImean for i in j]).mean() / np.array([i for j in FD_avg.RImean
#                                                                             for i in j]).mean())
#         ratio_RImax_mean_p.append(
#             np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_avg.RImax for
#                                                                            i in j]).mean())
#
#     # sensitity analysis: (no pooling & excluding at fd_rds = 0 : 0.1 : 0.5) vs (no pooling & no excluding)
#     fd_rds_avg = [rds_ / 100 for rds_ in list(range(0, 51, 1))]
#     ratio_FDD_mean_e, ratio_FDS_mean_e, ratio_number_e, ratio_number_NFD_e, ratio_RImean_mean_e, ratio_RImax_mean_e = [], [], [], [], [], []
#     for i in range(len(fd_pc_avg)):
#         FD_ = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
#                               pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
#                               eliminate_threshold=0.2,
#                               fd_pooling=False, fd_tc=fd_tc_avg, fd_pc=0.2, fd_excluding=True, fd_rds=fd_rds_avg[i])
#         ratio_FDD_mean_e.append(FD_.FDD_mean / FD_avg.FDD_mean)
#         ratio_FDS_mean_e.append(FD_.FDS_mean / FD_avg.FDS_mean)
#         ratio_number_e.append(sum(FD_.dp) / sum(FD_avg.dp))
#         ratio_number_NFD_e.append(FD_.dp.count(0) / FD_avg.dp.count(0))  # Number of Drought without flash drought
#         ratio_RImean_mean_e.append(
#             np.array([i for j in FD_.RImean for i in j]).mean() / np.array([i for j in FD_avg.RImean
#                                                                             for i in j]).mean())
#         ratio_RImax_mean_e.append(
#             np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_avg.RImax for
#                                                                            i in j]).mean())
#
#     # sensitity analysis: (pooling at fd_pc = 0.25 & excluding at fd_rds = 0 : 0.1 : 0.5) vs (no pooling & no excluding)
#     fd_pc_selected = 0.29
#     ratio_FDD_mean_pe, ratio_FDS_mean_pe, ratio_number_pe, ratio_number_NFD_pe, ratio_RImean_mean_pe, ratio_RImax_mean_pe = [], [], [], [], [], []
#     for i in range(len(fd_pc_avg)):
#         FD_ = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
#                               pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
#                               eliminate_threshold=0.2,
#                               fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_selected, fd_excluding=True, fd_rds=fd_rds_avg[i])
#         ratio_FDD_mean_pe.append(FD_.FDD_mean / FD_avg.FDD_mean)
#         ratio_FDS_mean_pe.append(FD_.FDS_mean / FD_avg.FDS_mean)
#         ratio_number_pe.append(sum(FD_.dp) / sum(FD_avg.dp))
#         ratio_number_NFD_pe.append(FD_.dp.count(0) / FD_avg.dp.count(0))  # Number of Drought without flash drought
#         ratio_RImean_mean_pe.append(np.array([i for j in FD_.RImean for i in j]).mean() / np.array(
#             [i for j in FD_avg.RImean for i in j]).mean())
#         ratio_RImax_mean_pe.append(
#             np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_avg.RImax for i in j]).mean())
#
#     # sensitity analysis: (pooling at fd_pc = 0.25 & excluding at fd_rds = 0 : 0.1 : 0.5) vs (pooling at fd_pc = 0.25
#     # & no excluding)
#     fd_pc_selected = 0.29
#     ratio_FDD_mean_pec, ratio_FDS_mean_pec, ratio_number_pec, ratio_number_NFD_pec, ratio_RImean_mean_pec, ratio_RImax_mean_pec = [], [], [], [], [], []
#     for i in range(len(fd_pc_avg)):
#         FD_ = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
#                               pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
#                               eliminate_threshold=0.2,
#                               fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_selected, fd_excluding=True, fd_rds=fd_rds_avg[i])
#         FD_compare = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True,
#                                      tc=tc_avg,
#                                      pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
#                                      eliminate_threshold=0.2,
#                                      fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_selected, fd_excluding=False,
#                                      fd_rds=fd_rds_avg[i])
#         ratio_FDD_mean_pec.append(FD_.FDD_mean / FD_compare.FDD_mean)
#         ratio_FDS_mean_pec.append(FD_.FDS_mean / FD_compare.FDS_mean)
#         ratio_number_pec.append(sum(FD_.dp) / sum(FD_compare.dp))
#         ratio_number_NFD_pec.append(FD_.dp.count(0) / FD_compare.dp.count(0))  # Number of Drought without flash drought
#         ratio_RImean_mean_pec.append(
#             np.array([i for j in FD_.RImean for i in j]).mean() / np.array([i for j in FD_compare.RImean
#                                                                             for i in j]).mean())
#         ratio_RImax_mean_pec.append(
#             np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_compare.RImax for
#                                                                            i in j]).mean())
#
#     # plot
#     fd_plot_subplot(x=fd_pc_avg, x_label="FD Pooling Ratio", ratio_FDD_mean=ratio_FDD_mean_p,
#                     ratio_FDS_mean=ratio_FDS_mean_p,
#                     ratio_number=ratio_number_p, ratio_number_NFD=ratio_number_NFD_p,
#                     ratio_RImean_mean=ratio_RImean_mean_p,
#                     ratio_RImax_mean=ratio_RImax_mean_p)
#
#     fd_plot_subplot(x=fd_pc_avg, x_label="FD Excluding Ratio", ratio_FDD_mean=ratio_FDD_mean_e,
#                     ratio_FDS_mean=ratio_FDS_mean_e,
#                     ratio_number=ratio_number_e, ratio_number_NFD=ratio_number_NFD_e,
#                     ratio_RImean_mean=ratio_RImean_mean_e,
#                     ratio_RImax_mean=ratio_RImax_mean_e)
#
#     fd_plot_subplot(x=fd_pc_avg, x_label="FD Excluding Ratio", ratio_FDD_mean=ratio_FDD_mean_pe,
#                     ratio_FDS_mean=ratio_FDS_mean_pe,
#                     ratio_number=ratio_number_pe, ratio_number_NFD=ratio_number_NFD_pe,
#                     ratio_RImean_mean=ratio_RImean_mean_pe,
#                     ratio_RImax_mean=ratio_RImax_mean_pe)
#
#     fd_plot_subplot(x=fd_pc_avg, x_label="FD Excluding Ratio", ratio_FDD_mean=ratio_FDD_mean_pec,
#                     ratio_FDS_mean=ratio_FDS_mean_pec,
#                     ratio_number=ratio_number_pec, ratio_number_NFD=ratio_number_NFD_pec,
#                     ratio_RImean_mean=ratio_RImean_mean_pec,
#                     ratio_RImax_mean=ratio_RImax_mean_pec)
#
#     out_ = np.array([ratio_FDD_mean_p, ratio_FDS_mean_p, ratio_number_p, ratio_number_NFD_p, ratio_RImean_mean_p,
#                      ratio_RImax_mean_p, ratio_FDD_mean_e, ratio_FDS_mean_e, ratio_number_e, ratio_number_NFD_e,
#                      ratio_RImean_mean_e, ratio_RImax_mean_e, ratio_FDD_mean_pe, ratio_FDS_mean_pe, ratio_number_pe,
#                      ratio_number_NFD_pe, ratio_RImean_mean_pe, ratio_RImax_mean_pe, ratio_FDD_mean_pec,
#                      ratio_FDS_mean_pec, ratio_number_pec, ratio_number_NFD_pec, ratio_RImean_mean_pec,
#                      ratio_RImax_mean_pec])
#     out = pd.DataFrame(out_, index=["ratio_FDD_mean_p", "ratio_FDS_mean_p", "ratio_number_p", "ratio_number_NFD_p",
#                                     "ratio_RImean_mean_p", "ratio_RImax_mean_p", "ratio_FDD_mean_e", "ratio_FDS_mean_e",
#                                     "ratio_number_e", "ratio_number_NFD_e", "ratio_RImean_mean_e", "ratio_RImax_mean_e",
#                                     "ratio_FDD_mean_pe", "ratio_FDS_mean_pe", "ratio_number_pe", "ratio_number_NFD_pe",
#                                     "ratio_RImean_mean_pe", "ratio_RImax_mean_pe", "ratio_FDD_mean_pec",
#                                     "ratio_FDS_mean_pec", "ratio_number_pec", "ratio_number_NFD_pec",
#                                     "ratio_RImean_mean_pec", "ratio_RImax_mean_pec"],
#                                     columns=[_ / 100 for _ in list(range(0, 51, 1))])
#     return out
#
#
# def compare():
#     FD_before = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=False, tc=tc_avg, pc=0.2,
#                                 excluding=False, rds=0.41, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
#                                 fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
#     _, _, out_put_before, _ = FD_before.general_out()
#
#     # result:
#     # tc = 5 pc=0.28 rds = 0.22
#     # fd_tc = 2 fd_pc=0.29 fd_rds=0.28
#     FD_after = FlashDrought.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5, pc=0.28,
#                                excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
#                                fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
#     _, _, out_put_after, _ = FD_after.general_out()
#     return out_put_before, out_put_after
