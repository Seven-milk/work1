# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import numpy as np
import pandas as pd
import FDIP
import os
from matplotlib import pyplot as plt

home = "H:/research/flash_drough/"

sm_rz_pentad_avg = np.loadtxt(os.path.join(home, "sm_rz_pentad_avg.txt"))
date_pentad = np.loadtxt(os.path.join(home, "date_pentad.txt"), dtype="int")

# baseline: not activate pooling/excluding and fd pooling/excluding
tc_avg = 5
FD_avg = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=False, tc=tc_avg, pc=0.2,
                 excluding=False, rds=0.41, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                 fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)


# SM_percentile, RI, out_put, dp = FD_avg.general_out()

def sensitity_Drought():
    # sensitity analysis: Drought DD_mean DS_mean number SM_min_mean under different pooling/excluding parameters
    # sensitity analysis: activate pooling at pc = 0 : 0.1 : 1 vs not activate pooling (ratio_DD/DS.._mean_p[50:] to select)
    pc_avg = [pc_ / 100 for pc_ in list(range(0, 101, 1))]
    ratio_DD_mean_p, ratio_DS_mean_p, ratio_number_p, ratio_SM_min_mean_p = [], [], [], []
    for i in range(len(pc_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_avg[i], excluding=False, rds=0.41, RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2,
                      fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
        ratio_DD_mean_p.append(FD_.DD.mean() / FD_avg.DD.mean())
        ratio_DS_mean_p.append(FD_.DS.mean() / FD_avg.DS.mean())
        ratio_number_p.append(len(FD_.DD) / len(FD_avg.DD))
        ratio_SM_min_mean_p.append(FD_.SM_min.mean() / FD_avg.SM_min.mean())

    # sensitity analysis: activate excluding at rds = 0 : 0.1 : 1 vs not activate excluding
    rds_avg = [rds_ / 100 for rds_ in list(range(0, 101, 1))]
    ratio_DD_mean_e, ratio_DS_mean_e, ratio_number_e, ratio_SM_min_mean_e = [], [], [], []
    for i in range(len(rds_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=False, tc=tc_avg,
                      pc=0.2, excluding=True, rds=rds_avg[i], RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2,
                      fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
        ratio_DD_mean_e.append(FD_.DD.mean() / FD_avg.DD.mean())
        ratio_DS_mean_e.append(FD_.DS.mean() / FD_avg.DS.mean())
        ratio_number_e.append(len(FD_.DD) / len(FD_avg.DD))
        ratio_SM_min_mean_e.append(FD_.SM_min.mean() / FD_avg.SM_min.mean())

    # sensitity analysis: activate excluding and pooling at rds = 0 : 0.1 : 1 and pc=0.54 vs not activate excluding
    pc_selected = 0.54
    rds_avg = [rds_ / 100 for rds_ in list(range(0, 101, 1))]
    ratio_DD_mean_pe, ratio_DS_mean_pe, ratio_number_pe, ratio_SM_min_mean_pe = [], [], [], []
    for i in range(len(rds_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_selected, excluding=True, rds=rds_avg[i], RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2, fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
        ratio_DD_mean_pe.append(FD_.DD.mean() / FD_avg.DD.mean())
        ratio_DS_mean_pe.append(FD_.DS.mean() / FD_avg.DS.mean())
        ratio_number_pe.append(len(FD_.DD) / len(FD_avg.DD))
        ratio_SM_min_mean_pe.append(FD_.SM_min.mean() / FD_avg.SM_min.mean())

    # sensitity analysis: activate excluding and pooling at rds = 0 : 0.1 : 1 and pc=0.54 vs not activate excluding but
    # activate pooling at pc=0.54
    pc_selected = 0.54
    rds_avg = [rds_ / 100 for rds_ in list(range(0, 101, 1))]
    ratio_DD_mean_pec, ratio_DS_mean_pec, ratio_number_pec, ratio_SM_min_mean_pec = [], [], [], []
    for i in range(len(rds_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_selected, excluding=True, rds=rds_avg[i], RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2, fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
        FD_compare = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True,
                             tc=tc_avg,
                             pc=pc_selected, excluding=False, rds=rds_avg[i], RI_threshold=0.05, eliminating=True,
                             eliminate_threshold=0.2, fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False,
                             fd_rds=0.41)
        ratio_DD_mean_pec.append(FD_.DD.mean() / FD_compare.DD.mean())
        ratio_DS_mean_pec.append(FD_.DS.mean() / FD_compare.DS.mean())
        ratio_number_pec.append(len(FD_.DD) / len(FD_compare.DD))
        ratio_SM_min_mean_pec.append(FD_.SM_min.mean() / FD_compare.SM_min.mean())

    def plot_subplot(x: list, x_label: str, ratio_DD_mean: list, ratio_DS_mean: list, ratio_number: list,
                     ratio_SM_min_mean: list):
        """ plot drought properties changing with pc or rds(x)"""
        font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
        font_ticks = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
        font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.plot(x, ratio_DD_mean, linestyle="-", marker="o", markersize=3)
        plt.xlim(0, 1)
        plt.ylim(1, )
        plt.title("Duration", font_title)
        plt.ylabel("Ratio of Mean", font_label)
        plt.xlabel(x_label, font_label)
        plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
        plt.yticks(fontproperties=font_ticks)

        plt.subplot(2, 2, 2)
        plt.plot(x, ratio_DS_mean, linestyle="-", marker="o", markersize=3)
        plt.xlim(0, 1)
        plt.ylim(1, )
        plt.title("Severity", font_title)
        plt.ylabel("Ratio of Mean", font_label)
        plt.xlabel(x_label, font_label)
        plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
        plt.yticks(fontproperties=font_ticks)

        plt.subplot(2, 2, 3)
        plt.plot(x, ratio_number, linestyle="-", marker="o", markersize=3)
        plt.xlim(0, 1)
        plt.ylim(top=1)
        plt.title("Events number", font_title)
        plt.ylabel("Ratio of Mean", font_label)
        plt.xlabel(x_label, font_label)
        plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
        plt.yticks(fontproperties=font_ticks)

        plt.subplot(2, 2, 4)
        plt.plot(x, ratio_SM_min_mean, linestyle="-", marker="o", markersize=3)
        plt.xlim(0, 1)
        plt.ylim(top=1.02)
        plt.title("Peak Value", font_title)
        plt.ylabel("Ratio of Mean", font_label)
        plt.xlabel(x_label, font_label)
        plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
        plt.yticks(fontproperties=font_ticks)

        plt.subplots_adjust(hspace=0.35, wspace=0.2)
        plt.show()

    # plot
    plot_subplot(x=pc_avg, x_label="Pooling Ratio", ratio_DD_mean=ratio_DD_mean_p, ratio_DS_mean=ratio_DS_mean_p,
                 ratio_number=ratio_number_p, ratio_SM_min_mean=ratio_SM_min_mean_p)

    plot_subplot(x=pc_avg, x_label="Excluding Ratio", ratio_DD_mean=ratio_DD_mean_e, ratio_DS_mean=ratio_DS_mean_e,
                 ratio_number=ratio_number_e, ratio_SM_min_mean=ratio_SM_min_mean_e)

    plot_subplot(x=pc_avg, x_label="Excluding Ratio", ratio_DD_mean=ratio_DD_mean_pe, ratio_DS_mean=ratio_DS_mean_pe,
                 ratio_number=ratio_number_pe, ratio_SM_min_mean=ratio_SM_min_mean_pe)

    plot_subplot(x=pc_avg, x_label="Excluding Ratio", ratio_DD_mean=ratio_DD_mean_pec, ratio_DS_mean=ratio_DS_mean_pec,
                 ratio_number=ratio_number_pec, ratio_SM_min_mean=ratio_SM_min_mean_pec)

# sensitity analysis: Flash Drought FDD FDS dp RI_mean? under different pooling/excluding parameters
# (fd_tc, fd_pc, fd_rds)
