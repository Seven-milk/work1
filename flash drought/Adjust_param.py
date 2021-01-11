# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# adjustment pooling and excluding parameters: Drought and FD
# result:
# tc = 5 pc=0.28 rds = 0.22
# fd_tc = 2 fd_pc=0.29 fd_rds=0.28
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


# sensitity analysis: drought
def plot_subplot(x: list, x_label: str, ratio_DD_mean: list, ratio_DS_mean: list, ratio_number: list,
                 ratio_SM_min_mean: list):
    """ plot drought properties changing with pc or rds(x)"""
    font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    font_ticks = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(x, ratio_DD_mean, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(1, )
    plt.title("Duration", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(2, 2, 2)
    plt.plot(x, ratio_DS_mean, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(1, )
    plt.title("Severity", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(2, 2, 3)
    plt.plot(x, ratio_number, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(top=1)
    plt.title("Events number", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(2, 2, 4)
    plt.plot(x, ratio_SM_min_mean, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(top=1.02)
    plt.title("Peak Value", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplots_adjust(hspace=0.35, wspace=0.2)
    plt.show()


def sensitity_Drought():
    # sensitity analysis: Drought DD_mean DS_mean number SM_min_mean under different pooling/excluding parameters
    # sensitity analysis: (pooling at pc = 0 : 0.1 : 0.5 & no excluding) vs (no pooling & no excluding)
    pc_avg = [pc_ / 100 for pc_ in list(range(0, 51, 1))]
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

    # sensitity analysis: (no pooling & excluding at rds = 0 : 0.1 : 0.5) vs (no pooling & no excluding)
    rds_avg = [rds_ / 100 for rds_ in list(range(0, 51, 1))]
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

    # sensitity analysis: (pooling at pc = 0.28 & excluding at rds = 0 : 0.1 : 0.5) vs (no pooling & no excluding)
    pc_selected = 0.28
    rds_avg = [rds_ / 100 for rds_ in list(range(0, 51, 1))]
    ratio_DD_mean_pe, ratio_DS_mean_pe, ratio_number_pe, ratio_SM_min_mean_pe = [], [], [], []
    for i in range(len(rds_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_selected, excluding=True, rds=rds_avg[i], RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2, fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
        ratio_DD_mean_pe.append(FD_.DD.mean() / FD_avg.DD.mean())
        ratio_DS_mean_pe.append(FD_.DS.mean() / FD_avg.DS.mean())
        ratio_number_pe.append(len(FD_.DD) / len(FD_avg.DD))
        ratio_SM_min_mean_pe.append(FD_.SM_min.mean() / FD_avg.SM_min.mean())

    # sensitity analysis: (pooling at pc = 0.28 & excluding at rds = 0 : 0.1 : 0.5) vs (pooling at pc = 0.28
    # & no excluding)
    pc_selected = 0.28
    rds_avg = [rds_ / 100 for rds_ in list(range(0, 51, 1))]
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

    # plot
    plot_subplot(x=pc_avg, x_label="Pooling Ratio", ratio_DD_mean=ratio_DD_mean_p, ratio_DS_mean=ratio_DS_mean_p,
                 ratio_number=ratio_number_p, ratio_SM_min_mean=ratio_SM_min_mean_p)

    plot_subplot(x=pc_avg, x_label="Excluding Ratio", ratio_DD_mean=ratio_DD_mean_e, ratio_DS_mean=ratio_DS_mean_e,
                 ratio_number=ratio_number_e, ratio_SM_min_mean=ratio_SM_min_mean_e)

    plot_subplot(x=pc_avg, x_label="Excluding Ratio", ratio_DD_mean=ratio_DD_mean_pe, ratio_DS_mean=ratio_DS_mean_pe,
                 ratio_number=ratio_number_pe, ratio_SM_min_mean=ratio_SM_min_mean_pe)

    plot_subplot(x=pc_avg, x_label="Excluding Ratio", ratio_DD_mean=ratio_DD_mean_pec, ratio_DS_mean=ratio_DS_mean_pec,
                 ratio_number=ratio_number_pec, ratio_SM_min_mean=ratio_SM_min_mean_pec)

    out_ = np.array([ratio_DD_mean_p, ratio_DS_mean_p, ratio_number_p, ratio_SM_min_mean_p, ratio_DD_mean_e,
                     ratio_DS_mean_e, ratio_number_e, ratio_SM_min_mean_e, ratio_DD_mean_pec, ratio_DS_mean_pec,
                     ratio_number_pec, ratio_SM_min_mean_pec])
    out = pd.DataFrame(out_, index=["ratio_DD_mean_p", "ratio_DS_mean_p", "ratio_number_p", "ratio_SM_min_mean_p",
                                    "ratio_DD_mean_e", "ratio_DS_mean_e", "ratio_number_e", "ratio_SM_min_mean_e",
                                    "ratio_DD_mean_pec", "ratio_DS_mean_pec", "ratio_number_pec",
                                    "ratio_SM_min_mean_pec"], columns=[_ / 100 for _ in list(range(0, 51, 1))])
    return out


# sensitity analysis: drought
def fd_plot_subplot(x: list, x_label: str, ratio_FDD_mean: list, ratio_FDS_mean: list, ratio_number: list,
                    ratio_number_NFD: list, ratio_RImean_mean: list, ratio_RImax_mean: list):
    """ plot drought properties changing with pc or rds(x)"""
    font_label = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
    font_ticks = {'family': 'Times New Roman', 'weight': 'normal', 'size': 18}
    font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 20}
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(x, ratio_FDD_mean, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(1, )
    plt.title("FD Duration", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(3, 2, 2)
    plt.plot(x, ratio_FDS_mean, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(1, )
    plt.title("FD Severity", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(3, 2, 3)
    plt.plot(x, ratio_number, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(top=1)
    plt.title("FD Events number", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(3, 2, 4)
    plt.plot(x, ratio_number_NFD, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(0.5, 1.5)
    plt.title("NFD Events number", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(3, 2, 5)
    plt.plot(x, ratio_RImean_mean, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(top=1)
    plt.title("FD RI Mean", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplot(3, 2, 6)
    plt.plot(x, ratio_RImax_mean, linestyle="-", marker="o", markersize=3)
    plt.xlim(0, 0.5)
    # plt.ylim(1, )
    plt.title("FD RI Max", font_title)
    plt.ylabel("Ratio of Mean", font_label)
    plt.xlabel(x_label, font_label)
    plt.xticks(ticks=x[::10], labels=[str(x_) for x_ in x[::10]], fontproperties=font_ticks)
    plt.yticks(fontproperties=font_ticks)

    plt.subplots_adjust(hspace=0.55, wspace=0.2)
    plt.show()


def sensitity_FD():
    # sensitity analysis: under different pooling/excluding parameters (fd_tc, fd_pc, fd_rds)
    fd_tc_avg = 2
    pc_avg = 0.28
    rds_avg = 0.22  # has been pooling and excluding in drought perspective
    FD_avg = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                     pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
                     eliminate_threshold=0.2,
                     fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)

    # sensitity analysis: (pooling at fd_pc = 0 : 0.1 : 0.5 & no excluding) vs (no pooling & no excluding)
    fd_pc_avg = [pc_ / 100 for pc_ in list(range(0, 51, 1))]
    ratio_FDD_mean_p, ratio_FDS_mean_p, ratio_number_p, ratio_number_NFD_p, ratio_RImean_mean_p, ratio_RImax_mean_p = [], [], [], [], [], []
    for i in range(len(fd_pc_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2,
                      fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_avg[i], fd_excluding=False, fd_rds=0.41)
        ratio_FDD_mean_p.append(FD_.FDD_mean / FD_avg.FDD_mean)
        ratio_FDS_mean_p.append(FD_.FDS_mean / FD_avg.FDS_mean)
        ratio_number_p.append(sum(FD_.dp) / sum(FD_avg.dp))
        ratio_number_NFD_p.append(FD_.dp.count(0) / FD_avg.dp.count(0))  # Number of Drought without flash drought
        ratio_RImean_mean_p.append(
            np.array([i for j in FD_.RImean for i in j]).mean() / np.array([i for j in FD_avg.RImean
                                                                            for i in j]).mean())
        ratio_RImax_mean_p.append(
            np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_avg.RImax for
                                                                           i in j]).mean())

    # sensitity analysis: (no pooling & excluding at fd_rds = 0 : 0.1 : 0.5) vs (no pooling & no excluding)
    fd_rds_avg = [rds_ / 100 for rds_ in list(range(0, 51, 1))]
    ratio_FDD_mean_e, ratio_FDS_mean_e, ratio_number_e, ratio_number_NFD_e, ratio_RImean_mean_e, ratio_RImax_mean_e = [], [], [], [], [], []
    for i in range(len(fd_pc_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2,
                      fd_pooling=False, fd_tc=fd_tc_avg, fd_pc=0.2, fd_excluding=True, fd_rds=fd_rds_avg[i])
        ratio_FDD_mean_e.append(FD_.FDD_mean / FD_avg.FDD_mean)
        ratio_FDS_mean_e.append(FD_.FDS_mean / FD_avg.FDS_mean)
        ratio_number_e.append(sum(FD_.dp) / sum(FD_avg.dp))
        ratio_number_NFD_e.append(FD_.dp.count(0) / FD_avg.dp.count(0))  # Number of Drought without flash drought
        ratio_RImean_mean_e.append(
            np.array([i for j in FD_.RImean for i in j]).mean() / np.array([i for j in FD_avg.RImean
                                                                            for i in j]).mean())
        ratio_RImax_mean_e.append(
            np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_avg.RImax for
                                                                           i in j]).mean())

    # sensitity analysis: (pooling at fd_pc = 0.25 & excluding at fd_rds = 0 : 0.1 : 0.5) vs (no pooling & no excluding)
    fd_pc_selected = 0.29
    ratio_FDD_mean_pe, ratio_FDS_mean_pe, ratio_number_pe, ratio_number_NFD_pe, ratio_RImean_mean_pe, ratio_RImax_mean_pe = [], [], [], [], [], []
    for i in range(len(fd_pc_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2,
                      fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_selected, fd_excluding=True, fd_rds=fd_rds_avg[i])
        ratio_FDD_mean_pe.append(FD_.FDD_mean / FD_avg.FDD_mean)
        ratio_FDS_mean_pe.append(FD_.FDS_mean / FD_avg.FDS_mean)
        ratio_number_pe.append(sum(FD_.dp) / sum(FD_avg.dp))
        ratio_number_NFD_pe.append(FD_.dp.count(0) / FD_avg.dp.count(0))  # Number of Drought without flash drought
        ratio_RImean_mean_pe.append(np.array([i for j in FD_.RImean for i in j]).mean() / np.array(
            [i for j in FD_avg.RImean for i in j]).mean())
        ratio_RImax_mean_pe.append(
            np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_avg.RImax for i in j]).mean())

    # sensitity analysis: (pooling at fd_pc = 0.25 & excluding at fd_rds = 0 : 0.1 : 0.5) vs (pooling at fd_pc = 0.25
    # & no excluding)
    fd_pc_selected = 0.29
    ratio_FDD_mean_pec, ratio_FDS_mean_pec, ratio_number_pec, ratio_number_NFD_pec, ratio_RImean_mean_pec, ratio_RImax_mean_pec = [], [], [], [], [], []
    for i in range(len(fd_pc_avg)):
        FD_ = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=tc_avg,
                      pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
                      eliminate_threshold=0.2,
                      fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_selected, fd_excluding=True, fd_rds=fd_rds_avg[i])
        FD_compare = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True,
                             tc=tc_avg,
                             pc=pc_avg, excluding=True, rds=rds_avg, RI_threshold=0.05, eliminating=True,
                             eliminate_threshold=0.2,
                             fd_pooling=True, fd_tc=fd_tc_avg, fd_pc=fd_pc_selected, fd_excluding=False,
                             fd_rds=fd_rds_avg[i])
        ratio_FDD_mean_pec.append(FD_.FDD_mean / FD_compare.FDD_mean)
        ratio_FDS_mean_pec.append(FD_.FDS_mean / FD_compare.FDS_mean)
        ratio_number_pec.append(sum(FD_.dp) / sum(FD_compare.dp))
        ratio_number_NFD_pec.append(FD_.dp.count(0) / FD_compare.dp.count(0))  # Number of Drought without flash drought
        ratio_RImean_mean_pec.append(
            np.array([i for j in FD_.RImean for i in j]).mean() / np.array([i for j in FD_compare.RImean
                                                                            for i in j]).mean())
        ratio_RImax_mean_pec.append(
            np.array([i for j in FD_.RImax for i in j]).mean() / np.array([i for j in FD_compare.RImax for
                                                                           i in j]).mean())

    # plot
    fd_plot_subplot(x=fd_pc_avg, x_label="FD Pooling Ratio", ratio_FDD_mean=ratio_FDD_mean_p,
                    ratio_FDS_mean=ratio_FDS_mean_p,
                    ratio_number=ratio_number_p, ratio_number_NFD=ratio_number_NFD_p,
                    ratio_RImean_mean=ratio_RImean_mean_p,
                    ratio_RImax_mean=ratio_RImax_mean_p)

    fd_plot_subplot(x=fd_pc_avg, x_label="FD Excluding Ratio", ratio_FDD_mean=ratio_FDD_mean_e,
                    ratio_FDS_mean=ratio_FDS_mean_e,
                    ratio_number=ratio_number_e, ratio_number_NFD=ratio_number_NFD_e,
                    ratio_RImean_mean=ratio_RImean_mean_e,
                    ratio_RImax_mean=ratio_RImax_mean_e)

    fd_plot_subplot(x=fd_pc_avg, x_label="FD Excluding Ratio", ratio_FDD_mean=ratio_FDD_mean_pe,
                    ratio_FDS_mean=ratio_FDS_mean_pe,
                    ratio_number=ratio_number_pe, ratio_number_NFD=ratio_number_NFD_pe,
                    ratio_RImean_mean=ratio_RImean_mean_pe,
                    ratio_RImax_mean=ratio_RImax_mean_pe)

    fd_plot_subplot(x=fd_pc_avg, x_label="FD Excluding Ratio", ratio_FDD_mean=ratio_FDD_mean_pec,
                    ratio_FDS_mean=ratio_FDS_mean_pec,
                    ratio_number=ratio_number_pec, ratio_number_NFD=ratio_number_NFD_pec,
                    ratio_RImean_mean=ratio_RImean_mean_pec,
                    ratio_RImax_mean=ratio_RImax_mean_pec)

    out_ = np.array([ratio_FDD_mean_p, ratio_FDS_mean_p, ratio_number_p, ratio_number_NFD_p, ratio_RImean_mean_p,
                     ratio_RImax_mean_p, ratio_FDD_mean_e, ratio_FDS_mean_e, ratio_number_e, ratio_number_NFD_e,
                     ratio_RImean_mean_e, ratio_RImax_mean_e, ratio_FDD_mean_pe, ratio_FDS_mean_pe, ratio_number_pe,
                     ratio_number_NFD_pe, ratio_RImean_mean_pe, ratio_RImax_mean_pe, ratio_FDD_mean_pec,
                     ratio_FDS_mean_pec, ratio_number_pec, ratio_number_NFD_pec, ratio_RImean_mean_pec,
                     ratio_RImax_mean_pec])
    out = pd.DataFrame(out_, index=["ratio_FDD_mean_p", "ratio_FDS_mean_p", "ratio_number_p", "ratio_number_NFD_p",
                                    "ratio_RImean_mean_p", "ratio_RImax_mean_p", "ratio_FDD_mean_e", "ratio_FDS_mean_e",
                                    "ratio_number_e", "ratio_number_NFD_e", "ratio_RImean_mean_e", "ratio_RImax_mean_e",
                                    "ratio_FDD_mean_pe", "ratio_FDS_mean_pe", "ratio_number_pe", "ratio_number_NFD_pe",
                                    "ratio_RImean_mean_pe", "ratio_RImax_mean_pe", "ratio_FDD_mean_pec",
                                    "ratio_FDS_mean_pec", "ratio_number_pec", "ratio_number_NFD_pec",
                                    "ratio_RImean_mean_pec", "ratio_RImax_mean_pec"],
                                    columns=[_ / 100 for _ in list(range(0, 51, 1))])
    return out


def compare():
    FD_before = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=False, tc=tc_avg, pc=0.2,
                     excluding=False, rds=0.41, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                     fd_pooling=False, fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)
    _, _, out_put_before, _ = FD_before.general_out()

    # result:
    # tc = 5 pc=0.28 rds = 0.22
    # fd_tc = 2 fd_pc=0.29 fd_rds=0.28
    FD_after = FDIP.FD(sm_rz_pentad_avg, Date_tick=date_pentad, timestep=73, threshold=0.4, pooling=True, tc=5, pc=0.28,
                     excluding=True, rds=0.22, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2,
                     fd_pooling=True, fd_tc=2, fd_pc=0.29, fd_excluding=True, fd_rds=0.28)
    _, _, out_put_after, _ = FD_after.general_out()
    return out_put_before, out_put_after
