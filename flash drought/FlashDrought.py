# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Flash Drought Identify

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Drought import Drought
import draw_plot
import useful_func
import os


class FlashDrought(Drought):
    ''' Flash Drought Class '''
    def __init__(self, drought_index, Date_tick, threshold=0.4, pooling=True, tc=1, pc=0.2, excluding=True,
                 rds=0.41, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2, fd_pooling=True, fd_tc=1,
                 fd_pc=0.2, fd_excluding=False, fd_rds=0.41):
        """
        Identify flash drought based on rule: extract from a drought event(start-1 : end)
        (ps: start/i - 1 means it can represent the rapid change from wet to drought)

        flash intensification : instantaneous RI[i]=drought_index[i] - drought_index[i-1] > RI_threshold ->
                                fd_flag_start, fd_flag_end

        note: this identification regard flash drought as a flash develop period of a normal drought event, and a normal
        drought can contain more than one flash drought
        note: generally, we do not consider excluding in FD, there have many condition to limit FD, keep it's "flash"
            and "drought"

        input: similar with Drought
            RI_threshold: the threshold of RI to extract extract flash development period(RI instantaneous)

            fd_pooling: bool, whether activate pooling: pooling flash drought in every drought events while
                        (fd_ti < fd_tc) & (fd_vi/fd_si < fd_pc) , based on IC method
                fd_tc:  predefined critical flash drought duration
                fd_pc: pooling ratio, the critical ratio of excess volume(vj) of inter-event time and the preceding deficit
                        volume(sj), this volume is the changed volume
            fd_excluding: bool, whether activate excluding: excluding while (fd_rd = FDDi / FDD_mean < fd_rds) or (fd_rs =
                        FDSi / FDS_mean < rds), based on IC method
                fd_rds: predefined critical excluding ratio, compare with fd_rd/rs = FDD/FDS / FDD/FDS_mean

            eliminating: bool, eliminate flash drought whose minimal SM_percentile > eliminate_threshold


        output: similar with Drought
            fd_flag_start, fd_flag_end, RImean, RImax: character of each drought events's flash develop periods, shrinkable start point
            plot and output have been override to plot and output flash drought

        """
        # inherit Drought
        super(FlashDrought, self).__init__(drought_index, Date_tick, threshold, pooling, tc, pc, excluding, rds)

        # general set
        self.RI_threshold = RI_threshold
        self.fd_pooling = fd_pooling
        self.fd_tc = fd_tc
        self.fd_pc = fd_pc
        self.fd_excluding = fd_excluding
        self.fd_rds = fd_rds
        self.eliminating = eliminating
        self.eliminate_threshold = eliminate_threshold

        # flash drought identification: extract flash develop period from drought events
        self.fd_flag_start, self.fd_flag_end, self.RI = self.develop_period()

        # pooling
        if self.fd_pooling:
            self.fd_cal_pooling()

        # eliminating
        if self.eliminating:
            self.fd_eliminate()

        # character
        self.dp, self.FDD, self.FDS, self.RImean, self.RImax = self.fd_character()
        self.FDD_mean = np.array([i for j in self.FDD for i in j]).mean()
        self.FDS_mean = np.array([i for j in self.FDS for i in j]).mean()

        # excluding, should be after character
        if self.fd_excluding:
            self.fd_cal_excluding()

    @staticmethod
    def fd_run_threshold(index: np.ndarray, threshold: float) -> (np.ndarray, np.ndarray):
        """ run_threshold to identify develop period (index > threshold, different with run_threshold)
            index: the base index
            threshold: the threshold to identify develop period(index > threshold)
            point explain(discrete): start > threshold, end > threshold --> it is shrinkable and strict
        """
        # define develop period based on index and threshold
        dry_flag = np.argwhere(index >= threshold).flatten()
        dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()
        dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()

        return dry_flag_start, dry_flag_end

    def develop_period(self) -> (list, list, list):
        """ flash drought identification

            method : extract from a drought event(start-1 : end)
            flash intensification : instantaneous RI[i] = drought_index[i] - drought_index[i-1]
                                    > RI_threshold -> fd_flag_start, fd_flag_end
            (note: start/i - 1 means it can represent the rapid change from wet to drought)

            unit = 1/time interval of input SM

            return:
                fd_flag_start, fd_flag_end, RImean, RImax for each flash drought event of every drought events
                structure:
                    list[array, array, ...], list...:
                        n list: n droughts
                        m array: m flash droughts in a drought
                    detail:
                        [--drought1[--flash develop period1[--fd_flag_start, fd_flag_end, RImean, RImax--], ..., --]

                RI = self.drought_index - np.append(self.drought_index[0], self.drought_index[:-1]) * -1 (series)

            note: distinguish the drought start flag, flash drought start flag, RI start flag
                flash drought start flag: compute from -1, it means the develop period from no-drought to drought
                    R[0] > RI_threshold -> R[0] = drought_index[0] - drought_index[-1], flag_start = 0
                RI start flag: the compute flag to evaluate RI condition, RI[i] = drought_index[i] - drought[i-1], means
                    change from the last point to this point
                drought start flag: it means the point has been achieved the threshold condition (match the shrinkable
                    run threshold and RI(SM(t)-SM(t-1))(same length with RI) condition)

                overall,
                    drought start flag = RI start flag(compute flag) = flash drought start flag + 1
                    drought start flag: satisfy drought condition
                    drought end flag: satisfy drought_index[end] < threshold
                    RI start flag(compute flag): RI[i] = drought_index[i] - drought[i-1], means change from the last
                                                point to this point
                    flash drought start flag: RI satisfy FD condition, it means change from flag - 1 point
                    flash end flag = RI end flag: satisfy RI[end] > RI_threshold = drought_index[i] - drought_index[i-1]

        """
        # the number of drought events
        n = len(self.dry_flag_start)

        # RI, hypothesize RI in the first position is zero(self.SM_percentile[0]-self.SM_percentile[0])
        # diff= sm[t]-sm[t-1] < 0: develop: RI > 0 ——> multiply factor: -1
        RI = self.drought_index - np.append(self.drought_index[0], self.drought_index[:-1])
        RI *= -1  # positive RI means drought_index decrease

        # list[array, array, ...] --> list: n(drought) array: m(flash)
        # each element in the list is a df_flag_start/end/RImean/RImax series of a drought event, namely, it represents
        # the develop periods(number > 1) of a drought event
        fd_flag_start, fd_flag_end = [], []

        # extract drought/flash develop period from each drought using RI
        for i in range(n):
            # the drought event from start to end
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]
            RI_ = RI[start: end + 1]

            # this flag based on RI_ index
            fd_flag_start_, fd_flag_end_ = self.fd_run_threshold(RI_, self.RI_threshold)
            # revert flag to make it based on drought_index index
            fd_flag_start_, fd_flag_end_ = fd_flag_start_ + start, fd_flag_end_ + start

            # append flash drought into drought list
            fd_flag_start.append(fd_flag_start_)  # list[array]
            fd_flag_end.append(fd_flag_end_)  # list[array]

        return fd_flag_start, fd_flag_end, RI

    def fd_cal_pooling(self):
        """ pooling flash drought in every drought event, based on IC method

            pooling condition: (fd_ti < fd_tc) & (fd_vi/fd_si < fd_pc)

        """
        # estimate every flash droughts
        for i in range(len(self.fd_flag_start)):
            size = len(self.fd_flag_start[i])
            j = 0
            # Loop for pooling
            while j < size - 1:
                start0 = self.fd_flag_start[i][j]
                end0 = self.fd_flag_end[i][j]
                start1 = self.fd_flag_start[i][j + 1]
                tj = start1 - end0
                vj = self.drought_index[start1] - self.drought_index[end0]
                sj = -(self.drought_index[end0] - self.drought_index[start0 - 1])  # start0 != 0, cause RI[0] = 0
                if (tj <= self.fd_tc) and ((vj / sj) <= self.fd_pc):
                    self.fd_flag_end[i][j] = self.fd_flag_end[i][j + 1]
                    self.fd_flag_start[i] = np.delete(self.fd_flag_start[i], j + 1)
                    self.fd_flag_end[i] = np.delete(self.fd_flag_end[i], j + 1)
                    size -= 1
                else:
                    j += 1

    def fd_eliminate(self):
        """ eliminate to satisfy the drought condition in flash drought identification

            eliminate condition: flash drought with minimal drought_index > eliminate_threshold

        """
        for i in range(len(self.fd_flag_start)):
            size = len(self.fd_flag_start[i])
            j = 0
            while j < size:
                flash_drought_min = min(self.drought_index[self.fd_flag_start[i][j]: self.fd_flag_end[i][j] + 1])
                if flash_drought_min > self.eliminate_threshold:
                    self.fd_flag_start[i] = np.delete(self.fd_flag_start[i], j)
                    self.fd_flag_end[i] = np.delete(self.fd_flag_end[i], j)
                    size -= 1
                else:
                    j += 1

    def fd_character(self) -> (list, list, list):
        """ extract the flash drought character variables

            dp: the develop period number of each drought
            FDD: duration of a flash drought event(develop period)
            FDS: severity of a flash drought event, equaling to the change between drought index at the flash drought
                start flag and min value of drought index

        """
        # general set
        n = len(self.dry_flag_start)  # the number of drought events
        dp = []  # number of develop periods of each drought event
        FDD, FDS, RImean, RImax = [], [], [], []

        # extract drought fd_character
        for i in range(n):

            # FDD/FDS/RI_mean/max set
            m = len(self.fd_flag_start[i])  # number of flash develop periods of each drought event
            dp.append(m)
            FDD_ = []
            FDS_ = []
            RImean_ = []
            RImax_ = []

            # extract
            for j in range(m):
                start = self.fd_flag_start[i][j]
                end = self.fd_flag_end[i][j]
                FDD_.append(end - start + 1)
                # end + 1 to contain the end point, start - 1 to present the pre point
                FDS_.append(min(self.drought_index[start: end + 1]) - self.drought_index[start - 1])

                # end + 1: RI[end] = drought_index[end] - drought_index[end-1] > RI_threshold, to contain the end point,
                # use [: end + 1]
                RImean_.append(self.RI[start: end + 1].mean())
                RImax_.append(self.RI[start: end + 1].max())

            # append to result
            FDD.append(FDD_)  # list[array]
            FDS.append(FDS_)
            RImean.append(RImean_)
            RImax.append(RImax_)

        return dp, FDD, FDS, RImean, RImax

    def fd_cal_excluding(self):
        """ excluding flash droughts for each flash droughts, based on IC method

            excluding condition: (fd_rd = FDDi / FDD_mean < fd_rds) or (fd_rs = FDSi / FDS_mean < rds)

        """
        # estimate every flash droughts
        for i in range(len(self.fd_flag_start)):
            size = len(self.fd_flag_start[i])
            j = 0
            # Loop for excluding
            while j < size:
                FD_RD = self.FDD[i][j] / self.FDD_mean
                FD_RS = self.FDS[i][j] / self.FDS_mean
                if (FD_RD <= self.fd_rds) or (FD_RS <= self.fd_rds):
                    self.fd_flag_start[i] = np.delete(self.fd_flag_start[i], j)
                    self.fd_flag_end[i] = np.delete(self.fd_flag_end[i], j)
                    self.RImean[i] = np.delete(self.RImean[i], j)
                    self.RImax[i] = np.delete(self.RImax[i], j)
                    self.FDD[i] = np.delete(self.FDD[i], j)
                    self.FDS[i] = np.delete(self.FDS[i], j)
                    self.dp[i] -= 1
                    size -= 1
                else:
                    j += 1

        # calculate FDD_mean, FDS_mean again after excluding
        self.FDD_mean = np.array([i for j in self.FDD for i in j]).mean()
        self.FDS_mean = np.array([i for j in self.FDS for i in j]).mean()

    def out_put(self, xlsx_on=False) -> pd.DataFrame:
        """ output the drought event to a dataframe and .xlsx file(to set xlsx_on=True) """

        # the number of flash drought events
        n = len(self.dry_flag_start)

        # calculate other characters which will be put into output
        Date_start = np.array([self.Date_tick[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date_tick[self.dry_flag_end[i]] for i in range(n)])
        threshold = np.full((n,), self.threshold, dtype='float')
        eliminate_threshold = np.full((n,), self.eliminate_threshold, dtype='float')
        RI_threshold = np.full((n,), self.RI_threshold, dtype='float')

        # build output character
        Drought_character = pd.DataFrame(
            np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.index_min,
                       self.index_min_flag, threshold, eliminate_threshold, RI_threshold)).T,
            columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "index_min", "index_min_flag",
                     "threshold", "eliminate_threshold", "RI_threshold"))

        Drought_character["fd_flag_start"] = self.fd_flag_start
        Drought_character["fd_flag_end"] = self.fd_flag_end
        Drought_character["RImean"] = self.RImean
        Drought_character["RImax"] = self.RImax
        Drought_character["develop period number"] = self.dp
        Drought_character["FDD"] = self.FDD
        Drought_character["FDS"] = self.FDS

        # save
        if xlsx_on:
            Drought_character.to_excel(f"Drought_FD_character_{xlsx_on}.xlsx", index=False)

        return Drought_character

    def plot(self, title="FD", save_on=False):
        """ plot the drought events: time series of index; threshold; drought events
        title: string, the title of this figure, it also will be the path for save figure
        yes: bool, save or not save this figure
        """
        # fig
        fig = draw_plot.Figure()
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="FD plot", labelx="Date", labely="Drought Index",
                              legend_on=True, xlim=(self.Date[0], self.Date[-1]))

        # drought index series
        drought_index_plot = draw_plot.PlotDraw(self.Date, self.drought_index, label="drought index",
                                                color="cornflowerblue", alpha=0.5, linewidth=0.5)
        draw.adddraw(drought_index_plot)

        # threshold series
        threshold_drought = np.full((len(self.Date),), fill_value=self.threshold)
        threshold_eliminate = np.full((len(self.Date),), fill_value=self.eliminate_threshold)
        threshold_drought_plot = draw_plot.PlotDraw(self.Date, threshold_drought, label=f"Threshold: {self.threshold}",
                                            color="chocolate", linewidth=1)
        threshold_eliminate_plot = draw_plot.PlotDraw(self.Date, threshold_eliminate, label=f"eliminate_threshold:"
                                            f" {self.eliminate_threshold}", color="darkred", linewidth=1)
        draw.adddraw(threshold_drought_plot)
        draw.adddraw(threshold_eliminate_plot)

        # trend series
        z = np.polyfit(range(len(self.Date)), self.drought_index, deg=1)
        p = np.poly1d(z)
        trend_series = p(range(len(self.Date)))
        trend_plot = draw_plot.PlotDraw(self.Date, trend_series, color="brown", alpha=0.5, label=f"Trend line")
        draw.adddraw(trend_plot)

        # drought
        for i in range(len(self.dry_flag_start)):
            # start and end
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]

            # event
            event_date = np.array(self.Date[start: end + 1], dtype=float)
            event = self.drought_index[start: end + 1]

            # event fix
            drought_index_start = self.drought_index[start]
            drought_index_end = self.drought_index[end]

            # drought_index_start/end must <= threshold
            if drought_index_start < self.threshold and start > 0:
                r = useful_func.intersection([start-1, self.threshold, start, self.threshold],
                                           [start-1, self.drought_index[start - 1], start, self.drought_index[start]])
                event_date = np.insert(event_date, 0, r[0])
                event = np.insert(event, 0, r[1])

            if drought_index_end < self.threshold and end < len(self.drought_index) - 1:
                r = useful_func.intersection([end, self.threshold, end + 1, self.threshold],
                                             [end, self.drought_index[end], end + 1, self.drought_index[end + 1]])
                event_date = np.append(event_date, r[0])
                event = np.append(event, r[1])

            # fill
            fig.ax.fill_between(event_date, event, self.threshold, alpha=1, facecolor="r", label="Drought events",
                            interpolate=True, zorder=20)

        # plot flash drought, flash drought events (from flash drought flag, -1: means change from no-drought)
        for i in range(len(self.fd_flag_start)):
            for j in range(len(self.fd_flag_start[i])):
                start = self.fd_flag_start[i][j] - 1  # RI = drought[i] - drought[i-1], -1 is used to present change
                end = self.fd_flag_end[i][j]
                fig.ax.plot(self.Date[start:end + 1], self.drought_index[start:end + 1], color="purple", linestyle='--',
                            linewidth=0.5, marker=7, markersize=2, zorder=30)
                print(f"drought: {i}", ";", f"flash develop period: {j}")

        # set ticks and labels
        fig.ax.set_xticks(self.Date[::int(len(self.Date) / 6)])  # set six ticks
        fig.ax.set_xticklabels(self.Date_tick[::int(len(self.Date) / 6)])
        fig.show()

        # save
        if save_on == True:
            plt.savefig("FD_" + title + ".tiff")

    def FD_character_plot(self, save_on=False):
        """ plot the boxplot of flash drought characters: RImean/RImax/FDD/FDS """
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.boxplot(np.array([i for j in self.RImean for i in j]))  # np.array([i for j in FD1.RImean for i in j]).min()
        plt.title("RImean")
        plt.subplot(2, 2, 2)
        plt.boxplot(np.array([i for j in self.RImax for i in j]))
        plt.title("RImax")
        plt.subplot(2, 2, 3)
        plt.boxplot(np.array([i for j in self.FDD for i in j]))
        plt.title("FDD")
        plt.subplot(2, 2, 4)
        plt.boxplot(np.array([i for j in self.FDS for i in j]))
        plt.title("FDS")

        # save
        if save_on == True:
            plt.savefig("FD_character_boxplot" + ".tiff")

    def general_out(self):
        # print set
        np.set_printoptions(threshold=np.inf)
        pd.set_option('display.width', None)

        # general out
        self.plot()
        self.Drought_character_plot()
        self.FD_character_plot()
        plt.show()
        # print("-----------------------------------------------")
        # print("drought index: \n", self.drought_index, "\n")
        # print("-----------------------------------------------")
        # print("RI: \n", self.RI, "\n")
        print("-----------------------------------------------")
        print("output: \n", self.out_put(), "\n")
        print("-----------------------------------------------")
        print("dp: \n", sum(self.dp), "\n")
        print("-----------------------------------------------")
        return self.RI, self.out_put(), self.dp


class FlashDrought_Frozen(FlashDrought):
    ''' Frozen Params
    tc = 6, pc = 0.5, rds = 0.2
    fd_tc = 2, fd_pc = 0.2
    '''
    def __init__(self, drought_index, Date_tick):
        super(FlashDrought_Frozen, self).__init__(drought_index, Date_tick, threshold=0.4, pooling=True, tc=6, pc=0.5,
                                                  excluding=True, rds=0.2, RI_threshold=0.05, eliminating=True,
                                                  eliminate_threshold=0.2, fd_pooling=True, fd_tc=2, fd_pc=0.2,
                                                  fd_excluding=False, fd_rds=0.41)


class FlashDrought_Liu(FlashDrought):
    ''' Reproduce YiLiu FlashDrought
    reference:
     Liu, Y., Zhu, Y., Zhang, L., Ren, L., Yuan, F., Yang, X., Jiang, S., 2020. Flash droughts characterization over
     China: From a perspective of the rapid intensification rate. Sci. Total Environ. 704, 135373.
     https://doi.org/10.1016/j.scitotenv.2019.135373
    '''

    def __init__(self, drought_index, Date_tick, threshold=0.4, pooling=False, tc=1, pc=0.2, excluding=False,
                 rds=0.41, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2, fd_pooling=False, fd_tc=1,
                 fd_pc=0.2, fd_excluding=False, fd_rds=0.41):
        super(FlashDrought_Liu, self).__init__(drought_index, Date_tick, threshold, pooling, tc, pc, excluding, rds,
                                               RI_threshold, eliminating, eliminate_threshold, fd_pooling, fd_tc,
                                               fd_pc, fd_excluding, fd_rds)

    def develop_period(self) -> (list, list, list):
        ''' overrides FlashDrought.develop_period '''
        # the number of drought events
        n = len(self.dry_flag_start)

        # RI, hypothesize RI in the first position is zero(self.SM_percentile[0]-self.SM_percentile[0])
        # diff= sm[t]-sm[t-1] < 0: develop: RI > 0 ——> multiply factor: -1
        RI = self.drought_index - np.append(self.drought_index[0], self.drought_index[:-1])
        RI *= -1

        # list[array, array, ...] --> list: n(drought) array: m(flash)
        # each element in the list is a df_flag_start/end/RImean/RImax series of a drought event, namely, it represents
        # the develop periods(number > 1) of a drought event
        fd_flag_start, fd_flag_end = [], []

        # extract drought/flash develop period from each drought using RI
        for i in range(n):
            # the drought event from start to end
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]
            peak = self.index_min_flag[i]
            RI_ = RI[start: peak + 1]
            if RI_.mean() > self.RI_threshold:
                fd_flag_start.append(np.array([start], dtype=int))
                fd_flag_end.append(np.array([peak], dtype=int))
            else:
                fd_flag_start.append(np.array([], dtype=int))
                fd_flag_end.append(np.array([], dtype=int))

        return fd_flag_start, fd_flag_end, RI


class FlashDrought_Liu_Frozen(FlashDrought_Liu):
    ''' Frozen Params
    pooling = False
    excluding = False
    RI_threshold = 0.05
    eliminating = True
    eliminate_threshold = 0.2
    '''
    def __init__(self, drought_index, Date_tick):
        super(FlashDrought_Liu_Frozen, self).__init__(drought_index, Date_tick, threshold=0.4, pooling=False, tc=1,
                                                      pc=0.2, excluding=False, rds=0.41, RI_threshold=0.05,
                                                      eliminating=True, eliminate_threshold=0.2, fd_pooling=False,
                                                      fd_tc=1, fd_pc=0.2, fd_excluding=False, fd_rds=0.41)


def testFlashDrought():
    FD = FlashDrought(drought_index, Date_tick=Date_tick, tc=tc, pc=pc, rds=rds, eliminating=True, fd_pooling=True,
                      fd_tc=fd_tc, fd_pc=fd_pc, fd_excluding=False)
    RI1, out_put1, dp1 = FD.general_out()


def testFlashDrought_Liu():
    Date_tick = []
    FD = FlashDrought_Liu(drought_index, Date_tick=Date_tick, pooling=False, excluding=False, tc=tc, pc=pc, rds=rds,
                          eliminating=True)
    RI2, out_put2, dp2 = FD.general_out()


def compareWithDifferentConfiguration():
    # ----------------------- compare results with different configuration -----------------------
    # # FD1
    # FD1 = FlashDrought(drought_index, Date_tick=[], tc=tc, pc=pc, rds=rds)
    # FD1.plot()
    # out1 = FD1.out_put()
    # # FD2
    # FD2 = FlashDrought(drought_index, Date_tick=[], pooling=False, excluding=False, tc=tc, pc=pc, rds=rds)
    # FD2.plot()
    # out2 = FD2.out_put()
    # # FD3
    # FD3 = FlashDrought(drought_index, Date_tick=[], pooling=True, excluding=False, tc=tc, pc=pc, rds=rds)
    # FD3.plot()
    # out3 = FD3.out_put()
    # # FD4
    # FD4 = FlashDrought(drought_index, Date_tick=[], pooling=False, excluding=True, tc=tc, pc=pc, rds=rds)
    # FD4.plot()
    # out4 = FD4.out_put()
    # ----------------------- compare results with pooling and excluding before and after -----------------------
    # path
    root = "H"
    home = f"{root}:/research/flash_drough/"
    sm_percentile_region_mean_path =\
        os.path.join(home, "GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile_RegionMean.npy")

    # read data
    sm_percentile_region_mean = np.load(sm_percentile_region_mean_path)
    drought_index = sm_percentile_region_mean[:, 1]

    # date
    date_pentad = pd.date_range('19480103', '20141231', freq='5d').strftime("%Y").to_numpy(dtype="int")

    # before
    fd_before = FlashDrought(drought_index, date_pentad, threshold=0.4, pooling=False, tc=1, pc=0.2, excluding=False,
                 rds=0.41, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2, fd_pooling=False, fd_tc=1,
                 fd_pc=0.2, fd_excluding=False, fd_rds=0.41)

    # after
    fd_after = FlashDrought_Frozen(drought_index, Date_tick=date_pentad)

    # out
    fd_before.general_out()
    fd_after.general_out()

    # save
    fd_before.out_put(xlsx_on="before")
    fd_after.out_put(xlsx_on="after")


if __name__ == "__main__":
    # test code through using a random series(with a random seed)
    # ----------------------- General set -----------------------
    # np.random.seed(15)
    drought_index = np.random.rand(365 * 3)
    drought_index = np.convolve(drought_index, np.repeat(1 / 2, 3), mode='full')  # running means
    tc = 10
    pc = 0.5
    rds = 0.41
    fd_pc = 0.2
    fd_tc = 2
    Date_tick = []
    # ----------------------- FlashDrought -----------------------
    # testFlashDrought()

    # ----------------------- FlashDrought_Liu -----------------------
    # testFlashDrought_Liu()

    # ----------------------- compare results with different configuration -----------------------
    compareWithDifferentConfiguration()
