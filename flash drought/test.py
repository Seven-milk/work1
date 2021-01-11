# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
# Flash Drought Identify Process
# TODO ADD reference

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class SM_percentile():
    def __init__(self, SM, timestep):
        """
        SM ——> SM_percentile: Statistics method(self.cal_SM_percentile)
        input:
            SM: SOIL MOISTURE, list or numpy array
            timestep: the timestep of SM （the number of SM data in one year）, which is used to do cal_SM_percentile
            (reshape a vector to a array, which shape is (n(year)+1 * timestep))
                365 : daily, 365 data in one year
                12 : monthly, 12 data in one year
                73 : pentad(5), 73 data in one year
                x ：x data in one year

        output:
            self.SM_percentile
        """
        self.SM_percentile = self.cal_SM_percentile()

    @staticmethod
    def percentile(x: np.ndarray, method="kde", path="1", y=0, bw_method="scott") -> np.ndarray:
        """ calculate the percentile for each point in x, there are two method to estimate the distribution
        method1: kernel density estimation, method='kde', bw_method='Scott'
        method2: Gringorten estimation, method='Gringorten'
        input:
            x: 1D numpy.ndarray
        output:
            x_percentile: 1D numpy.ndarray
        """
        if method == "kde":
            kde = stats.gaussian_kde(x, bw_method=bw_method)
            x_percentile = np.array([kde.integrate_box_1d(low=0, high=x[i]) for i in range(len(x))])
            # plot while y set to 1
            if y == 1:
                fig, ax1 = plt.subplots()
                ax2 = ax1.twinx()
                ax1.hist(x, bins=int(len(x) * kde.covariance_factor()), label="Hist", alpha=0.5)
                x_eval = np.linspace(x.min(), x.max(), num=int((x.max() - x.min()) * 100))
                ax1.plot(x, np.zeros(x.shape), '+', color='navy', ms=20, label="Samples")
                ax1.set_ylabel("Number of samples")
                ax2.plot(x_eval, kde(x_eval), 'r-', label="KDE based on bw_method: " + bw_method)
                ax2.set_ylabel("PDF")
                ax1.set_title("Kernel density estimation")
                ax1.set_xlim(x.min(), x.max())
                plt.legend([ax1, ax2, ])
                handles1, labels1 = ax1.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
                plt.savefig(f"/SM_kde/SM_kde{path}")
        elif method == "Gringorten":
            series_ = pd.Series(x)
            x_percentile_ = [(series_.rank(axis=0, method="min", ascending=1)[i] - 0.44) / (len(series_) + 0.12) for i
                             in range(len(series_))]
            x_percentile = np.array(x_percentile_)
        return x_percentile

    def cal_SM_percentile(self, y=0) -> np.ndarray:
        """ calculate SM percentile using SM, with process of reshape based on timestep(e.g. daily)
        ps: distribution fited in a section
        SM 1D ——> timestep * section 2D ——> SM_percentile 1D
        """
        n = len(self.SM) // self.timestep
        SM_percentile = np.full((n + 1, self.timestep), np.NAN, dtype='float')
        SM_percentile[:n, :] = self.SM[:(n * self.timestep)].reshape((n, self.timestep))
        SM_percentile[n:, :len(self.SM[(n * self.timestep):])] = self.SM[(n * self.timestep):]
        for i in range(self.timestep):
            l = SM_percentile[:, i]
            l = l[~np.isnan(l)]
            # percentile
            l = self.percentile(l, method="kde", path=str(i), y=y)
            SM_percentile[:len(l), i] = l
        SM_percentile = SM_percentile.flatten()
        SM_percentile = SM_percentile[~np.isnan(SM_percentile)]
        return SM_percentile


class Drought(SM_percentile):
    def __init__(self, SM, timestep=365, Date_tick=[], threshold=0.4, pooling=True, tc=1, pc=0.2, excluding=True,
                 rds=0.41):
        """
        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold: the threshold to identify dry bell
            pooling: bool, whether activate pooling: pooing while (ti < tc) & (vi/si < pc), based on IC method
                tc: predefined critical duration
                pc: pooling ratio, the critical ratio of excess volume(vi) of inter-event time and the preceding deficit
                    volume(si)
            excluding: bool, whether activate excluding: excluding while (rd = di / dmean < rds) or (rs = si / smean < rds)
                    , based on IC method
            Date_tick: the Date of SM, np.ndarray
            timestep: the timestep of SM （the number of SM data in one year）, which is used to do cal_SM_percentile
            (reshape a vector to a array, which shape is (n(year)+1 * timestep))
                365 : daily, 365 data in one year
                12 : monthly, 12 data in one year
                73 : pentad(5), 73 data in one year
                x ：x data in one year

        output:
            self.SM_percentile: SM_percentile calculated by SM using self.cal_SM_percentile
            self.dry_flag_start, self.dry_flag_end: start end of each drought events(shrinkable)
            self.DD, self.DS, self.SM_min: character of each drought events
            plot: use self.plot(plot(self, title="Drought", yes=0)), save figure set yes = 1: plot dorught
            xlsx: use self.out_put(self, xlsx=1) , set xlsx=1: out put drought the xlsx
        """
        if type(SM) is np.ndarray:
            self.SM = SM
        else:
            self.SM = np.array(SM, dtype='float')
        if len(Date_tick) == 0:
            self.Date_tick = list(range(len(SM)))
        else:
            self.Date_tick = Date_tick
        self.Date = list(range(len(SM)))
        self.timestep = timestep
        self.threshold = threshold
        self.pooling = pooling
        self.tc = tc
        self.pc = pc
        self.excluding = excluding
        self.rds = rds
        SM_percentile.__init__(self, SM, timestep)
        self.dry_flag_start, self.dry_flag_end = self.run_threshold(self.SM_percentile, self.threshold)
        if self.pooling:
            self.cal_pooling()
        self.DD, self.DS, self.SM_min, self.SM_min_flag = self.character()
        if self.excluding:
            self.cal_excluding()

    @staticmethod
    def run_threshold(index: np.ndarray, threshold: float) -> (np.ndarray, np.ndarray):
        """ run_threshold to identify dry bell (start-end)
        index: the base index
        threshold: the threshold to identify dry bell(index < threshold)
        point explain(discrete): start < threshold, end < threshold --> it is shrinkable and strict
        """
        # define drought based on index and threshold
        dry_flag = np.argwhere(index <= threshold).flatten()
        dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()
        dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()
        return dry_flag_start, dry_flag_end

    def cal_pooling(self):
        """ pooling while (ti < tc) & (vi/si < pc), based on IC method"""
        size = len(self.dry_flag_start)
        i = 0
        while i < size - 1:
            ti = self.dry_flag_start[i + 1] - self.dry_flag_end[i]
            vi = (self.SM_percentile[self.dry_flag_end[i] + 1: self.dry_flag_start[i + 1]] - self.threshold).sum()
            si = (self.threshold - self.SM_percentile[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]).sum()
            if (ti < self.tc) and ((vi / si) < self.pc):
                self.dry_flag_end[i] = self.dry_flag_end[i + 1]
                self.dry_flag_start = np.delete(self.dry_flag_start, i + 1)
                self.dry_flag_end = np.delete(self.dry_flag_end, i + 1)
                size -= 1
            else:
                i += 1

    def character(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """ extract the drought character variables
        DD: duration of drought events (unit based on timestep)
        DS: severity of drought events (based on SM_percentile)
        SM_min: the min value of SM (based on SM_percentile)
        """
        n = len(self.dry_flag_start)  # the number of flash drought events
        DD = self.dry_flag_end - self.dry_flag_start + 1
        x = self.threshold - self.SM_percentile
        DS = np.array([x[self.dry_flag_start[i]: self.dry_flag_end[i] + 1].sum() for i in range(n)], dtype='float')
        SM_min = np.array([min(self.SM_percentile[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) for i in range(n)]
                          , dtype='float')
        SM_min_flag = np.array([np.argmin(self.SM_percentile[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) +
                                self.dry_flag_start[i] for i in range(n)], dtype='float')
        return DD, DS, SM_min, SM_min_flag

    def cal_excluding(self):
        """ excluding while (rd = di / dmean < rds) or (rs = si / smean < rds), based on IC method """
        RD = self.DD / self.DD.mean()
        RS = self.DS / self.DS.mean()
        size = len(RD)
        i = 0
        while i < size:
            if (RD[i] < self.rds) or (RS[i] < self.rds):
                self.dry_flag_start = np.delete(self.dry_flag_start, i)
                self.dry_flag_end = np.delete(self.dry_flag_end, i)
                self.DD = np.delete(self.DD, i)
                self.DS = np.delete(self.DS, i)
                self.SM_min = np.delete(self.SM_min, i)
                self.SM_min_flag = np.delete(self.SM_min_flag, i)
                RD = np.delete(RD, i)
                RS = np.delete(RS, i)
                size -= 1
            else:
                i += 1

    def out_put(self, xlsx=0) -> pd.DataFrame:
        """ output the drought event to a dataframe and .xlsx file(to set xlsx=1) """
        n = len(self.dry_flag_start)  # the number of flash drought events
        Date_start = np.array([self.Date_tick[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date_tick[self.dry_flag_end[i]] for i in range(n)])
        threshold = np.full((n,), self.threshold, dtype='float')
        Drought_character = pd.DataFrame(
            np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.SM_min,
                       self.SM_min_flag, threshold)).T,
            columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "SM_min", "SM_min_flag"
                     , "thrshold"))
        if xlsx == 1:
            Drought_character.to_excel("/Drought_character", index=False)
        return Drought_character

    def plot(self, title="Drought", yes=0):
        """ plot the drought events: time series of index; threshold; drought events
        title: string, the title of this figure, it also will be the path for save figure
        yes: bool, save or not save this figure
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # plot fundamental figure : SM
        ax1.bar(self.Date, self.SM, label="observation SM", color="cornflowerblue", alpha=0.5)  # sm bar
        ax2.plot(self.Date, self.SM_percentile, label="SM Percentile", color="royalblue",
                 linewidth=0.9)  # sm_percentile
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=0.5), label=f"Mean=0.5", color="sandybrown")  # mean
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=self.threshold),
                 label=f"Threshold={self.threshold}", color="chocolate")  # threshold
        # plot the trend line of sm
        z = np.polyfit(range(len(self.Date)), self.SM, deg=1)
        p = np.poly1d(z)
        ax1.plot(self.Date, p(range(len(self.Date))), color="brown", alpha=0.5, label=f"Trend:{p}")
        # plot drought events
        events = np.full((len(self.Date),), fill_value=self.threshold)
        for i in range(len(self.dry_flag_start)):
            events[self.dry_flag_start[i]:self.dry_flag_end[i] + 1] = \
                self.SM_percentile[self.dry_flag_start[i]:self.dry_flag_end[i] + 1]
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]
            ax2.plot(self.Date[start:end + 1], self.SM_percentile[start:end + 1], "r", linewidth=1)
        ax2.fill_between(self.Date, events, self.threshold, alpha=0.8, facecolor="peru",
                         label="Drought events", interpolate=True)
        # set figure
        ax1.set_ylabel("SM")
        ax2.set_ylabel("SM Percentile")
        ax2.set_xlabel("Date")
        ax1.set_xlim(xmin=self.Date[0], xmax=self.Date[-1])
        ax2.set_xlim(xmin=self.Date[0], xmax=self.Date[-1])
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax1.set_title(title, loc="center")
        ax1.set_xticks(self.Date[::int(len(self.Date) / 6)])  # set six ticks
        ax1.set_xticklabels(self.Date_tick[::int(len(self.Date) / 6)])
        ax2.set_xticks(self.Date[::int(len(self.Date) / 6)])  # set six ticks
        ax2.set_xticklabels(self.Date_tick[::int(len(self.Date) / 6)])
        plt.show()
        if yes == 1:
            plt.savefig("Drought/" + title)

    def Drought_character_plot(self, yes=0):
        """ plot the boxplot of flash drought characters: RImean/RImax/FDD/FDS"""
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.boxplot(self.DD)  # np.array([i for j in FD1.RImean for i in j]).min()
        plt.title("DD")
        plt.subplot(2, 2, 2)
        plt.boxplot(self.DS)
        plt.title("DS")
        plt.subplot(2, 2, 3)
        plt.boxplot(self.SM_min)
        plt.title("SM_min")
        plt.subplot(2, 2, 4)
        plt.boxplot(self.SM_percentile)
        plt.title("SM_percentile")
        if yes == 1:
            plt.savefig("Drought_character_boxplot")


class FD(Drought):
    def __init__(self, SM, timestep=365, Date_tick=[], threshold=0.4, pooling=True, tc=1, pc=0.2, excluding=True,
                 rds=0.41, RI_threshold=0.05, eliminating=True, eliminate_threshold=0.2, fd_pooling=True, fd_tc=1,
                 fd_pc=0.2, fd_excluding=True, fd_rds=0.41):
        """
        Identify flash drought based on rule: extract from a drought event(start-1 : end)
        (ps: start/i - 1 means it can represent the rapid change from wet to drought)
            flash intensification : instantaneous RI[i] > RI_threshold -> fd_flag_start = i - 1, fd_flag_end
            drought(activate by eliminating = True) : eliminate flash drought whose minimal SM_percentile > eliminate_threshold

        fd_pooling: bool, whether activate pooling: pooling flash drought in every drought events while (fd_ti < fd_tc)
        & (fd_vi/fd_si < fd_pc) , based on IC method
            fd_tc:  predefined critical flash drought duration
            fd_pc: pooling ratio, the critical ratio of excess volume(vj) of inter-event time and the preceding deficit
                    volume(sj), this volume is the changed volume
        fd_excluding: bool, whether activate excluding: excluding while (fd_rd = FDDi / FDD_mean < fd_rds) or (fd_rs =
        FDSi / FDS_mean < rds), based on IC method
            fd_rds: predefined critical excluding ratio, compare with fd_rd/rs = FDD/FDS / FDD/FDS_mean
        this identification regard flash drought as a flash develop period of a normal drought event, and a normal
        drought can contain more than one flash drought

        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold: the threshold to identify dry bell
            pooling: bool, whether activate pooling: pooing while (ti < tc) & (vi/si < pc), based on IC method
                tc: predefined critical duration
                pc: pooling ratio, the critical ratio of excess volume(vi) of inter-event time and the preceding deficit
                    volume(si)
            excluding: bool, whether activate excluding: excluding while (rd = di / dmean < rds) or (rs = si / smean < rds)
                    , based on IC method
                rds: predefined critical excluding ratio, compare with rd/rs = d/s / d/s_mean
            RI_threshold: the threshold of RI to extract extract flash development period(RI instantaneous)
            Date_tick: the Date of SM, np.ndarray
            timestep: the timestep of SM （the number of SM data in one year）, which is used to do cal_SM_percentile
            (reshape a vector to a array, which shape is (n(year)+1 * timestep))
                365 : daily, 365 data in one year
                12 : monthly, 12 data in one year
                73 : pentad(5), 73 data in one year
                x ：x data in one year

        output:
            self.SM_percentile: SM_percentile calculated by SM using self.cal_SM_percentile
            self.dry_flag_start, self.dry_flag_end: start end of each drought events
            self.DD, self.DS, self.SM_min: character of each drought events
            fd_flag_start, fd_flag_end, RImean, RImax: character of each drought events's flash develop periods, shrinkable start point
            plot: use self.plot(plot(self, title="Drought", yes=0)), save figure set yes = 1: plot dorught, plot flash drought using compute start point[-1]
            xlsx: use self.out_put(self, xlsx=1) , set xlsx=1: out put drought the xlsx

        """
        Drought.__init__(self, SM, timestep, Date_tick, threshold, pooling, tc, pc, excluding, rds)
        self.RI_threshold = RI_threshold
        self.fd_pooling = fd_pooling
        self.fd_tc = fd_tc
        self.fd_pc = fd_pc
        self.fd_excluding = fd_excluding
        self.fd_rds = fd_rds
        self.eliminating = eliminating
        self.eliminate_threshold = eliminate_threshold
        self.fd_flag_start, self.fd_flag_end, self.RI = self.develop_period()
        if self.fd_pooling:
            self.fd_cal_pooling()
        if self.eliminating:
            self.fd_eliminate()
        self.dp, self.FDD, self.FDS, self.RImean, self.RImax = self.fd_character()
        self.FDD_mean = np.array([i for j in self.FDD for i in j]).mean()
        self.FDS_mean = np.array([i for j in self.FDS for i in j]).mean()
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

    def develop_period(self) -> (list, list, list, list):
        """ To sitisfy the flash intensification condition in the flash drought identification
        method : extract from a drought event(start-1 : end)
            flash intensification : instantaneous RI[i] > RI_threshold -> fd_flag_start = i - 1, fd_flag_end
            (ps: start/i - 1 means it can represent the rapid change from wet to drought)

        unit = 1/time interval of input SM

        return:
            fd_flag_start, fd_flag_end, RImean, RImax for each flash drought event of every drought events
                list[array, array, ...] --> list: n(drought) array: m(flash)
                SM[--drought1[--flash develop period1[--fd_flag_start, fd_flag_end, RImean, RImax--],..flash develop periodn[]--], ...droughtn[]--]

            RI = self.SM_percentile - np.append(self.SM_percentile[0], self.SM_percentile[:-1]) * -1(series rather events)

        """
        n = len(self.dry_flag_start)  # the number of drought events
        # distinguish the compute start point and flag start point:
        # compute start point: compute from -1, it means the develop period from no-drought to drought
        # flag start point: it means the point has been achieved the threshold condition (match the shrinkable
        # run threshold and RI(SM(t)-SM(t-1))(same length with RI))
        RI = self.SM_percentile - np.append(self.SM_percentile[0], self.SM_percentile[:-1])
        RI = RI * -1
        # calculate RI of the whole timeseries of SM_peritile, then extract drought/flash develop period from it,
        # hypothesize RI in the first position is zero(self.SM_percentile[0]-self.SM_percentile[0])
        # diff= sm[t]-sm[t-1] < 0: develop: RI > 0 ——> multiply factor: -1
        # list[array, array, ...] --> list: n(drought) array: m(flash)
        fd_flag_start, fd_flag_end = [], []
        # list, each element is a df_flag_start/end/RImean/RImax series of a drought event
        # namely, it represents the develop periods(number > 1) of a dought event

        # extract drought/flash develop period from each dorught using RI
        for i in range(n):
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]
            RI_ = RI[start: end + 1]
            fd_flag_start_, fd_flag_end_ = self.fd_run_threshold(RI_, self.RI_threshold)  # this flag based on RI_ index
            fd_flag_start_, fd_flag_end_ = fd_flag_start_ + start, fd_flag_end_ + start  # calculate flag based on SM index
            # ps: start / i - 1 means it can represent the rapid change from wet to drought, check the first point
            # (or index can be -1)
            if len(fd_flag_start_) > 0:
                if fd_flag_start_[0] == 0:
                    fd_flag_start_ -= 1
                    fd_flag_start_[0] = 0
                else:
                    fd_flag_start_ -= 1
            fd_flag_start.append(fd_flag_start_)  # list(np.ndarray)
            fd_flag_end.append(fd_flag_end_)  # list(np.ndarray)

        return fd_flag_start, fd_flag_end, RI

    def fd_cal_pooling(self):
        """ pooling flash drought in every drought events while (fd_ti < fd_tc) & (fd_vi/fd_si < fd_pc) , based on IC
        method """
        for i in range(len(self.fd_flag_start)):
            size = len(self.fd_flag_start[i])
            j = 0
            while j < size - 1:
                tj = self.fd_flag_start[i][j + 1] - self.fd_flag_end[i][j]
                vj = self.SM_percentile[self.fd_flag_start[i][j + 1]] - self.SM_percentile[self.fd_flag_end[i][j]]
                sj = -(self.SM_percentile[self.fd_flag_end[i][j]] - self.SM_percentile[self.fd_flag_start[i][j]])
                if (tj < self.fd_tc) and ((vj / sj) < self.fd_pc):
                    self.fd_flag_end[i][j] = self.fd_flag_end[i][j + 1]
                    self.fd_flag_start[i] = np.delete(self.fd_flag_start[i], j + 1)
                    self.fd_flag_end[i] = np.delete(self.fd_flag_end[i], j + 1)
                    size -= 1
                else:
                    j += 1

    def fd_eliminate(self):
        """ To satisfy the drought condition in the flash drought identification
        method : eliminate flash drought whose minimal SM_percentile > eliminate_threshold
        """
        for i in range(len(self.fd_flag_start)):
            size = len(self.fd_flag_start[i])
            j = 0
            while j < size:
                if min(self.SM_percentile[
                       self.fd_flag_start[i][j]: self.fd_flag_end[i][j] + 1]) > self.eliminate_threshold:
                    self.fd_flag_start[i] = np.delete(self.fd_flag_start[i], j)
                    self.fd_flag_end[i] = np.delete(self.fd_flag_end[i], j)
                    size -= 1
                else:
                    j += 1

    def fd_character(self) -> (list, list, list):
        """ extract the drought character variables
        dp: the develop period number of each drought
        FDD: duration of a flash drought event(develop period)
        FDS: severity of a flash drought event, equaling to the change during flash drought period
        """
        n = len(self.dry_flag_start)  # the number of drought events
        dp = []  # number of develop periods of each drought event
        FDD, FDS, RImean, RImax = [], [], [], []
        for i in range(n):
            # FDD/FDS/RI_mean/max
            m = len(self.fd_flag_start[i])  # number of flash develop periods of each drought event
            dp.append(m)
            FDD_ = []
            FDS_ = []
            RImean_ = []
            RImax_ = []
            for j in range(m):
                start = self.fd_flag_start[i][j]
                end = self.fd_flag_end[i][j]
                FDD_.append(end - start + 1)
                FDS_.append(self.SM_percentile[end] - self.SM_percentile[start])
                RImean_.append(self.RI[start + 1: end + 1].mean())  # start + 1: RI calculate from start + 1
                RImax_.append(self.RI[start + 1: end + 1].max())
            FDD.append(FDD_)  # list(np.ndarray)
            FDS.append(FDS_)
            RImean.append(RImean_)
            RImax.append(RImax_)
        return dp, FDD, FDS, RImean, RImax

    def fd_cal_excluding(self):
        """ excluding while (fd_rd = FDDi / FDD_mean < fd_rds) or (fd_rs = FDSi / FDS_mean < rds)
        , based on IC method """
        for i in range(len(self.fd_flag_start)):
            size = len(self.fd_flag_start[i])
            j = 0
            while j < size:
                FD_RD = self.FDD[i][j] / self.FDD_mean
                FD_RS = self.FDS[i][j] / self.FDS_mean
                if (FD_RD < self.fd_rds) or (FD_RS < self.fd_rds):
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
        self.FDD_mean = np.array([i for j in self.FDD for i in j]).mean()
        self.FDS_mean = np.array([i for j in self.FDS for i in j]).mean()

    def out_put(self, xlsx=0) -> pd.DataFrame:
        """ output the drought event to a dataframe and .xlsx file(to set xlsx=1) """
        n = len(self.dry_flag_start)  # the number of flash drought events
        Date_start = np.array([self.Date_tick[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date_tick[self.dry_flag_end[i]] for i in range(n)])
        threshold = np.full((n,), self.threshold, dtype='float')
        eliminate_threshold = np.full((n,), self.eliminate_threshold, dtype='float')
        RI_threshold = np.full((n,), self.RI_threshold, dtype='float')
        Drought_character = pd.DataFrame(
            np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.SM_min,
                       self.SM_min_flag, threshold, eliminate_threshold, RI_threshold)).T,
            columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "SM_min", "SM_min_flag",
                     "threshold", "eliminate_threshold", "RI_threshold"))
        Drought_character["fd_flag_start"] = self.fd_flag_start
        Drought_character["fd_flag_end"] = self.fd_flag_end
        Drought_character["RImean"] = self.RImean
        Drought_character["RImax"] = self.RImax
        Drought_character["number of dp"] = self.dp
        Drought_character["FDD"] = self.FDD
        Drought_character["FDS"] = self.FDS
        if xlsx == 1:
            Drought_character.to_excel("/Drought_character", index=False)
        return Drought_character

    def plot(self, title="Drought", yes=0):
        """ plot the drought events: time series of index; threshold; drought events
        title: string, the title of this figure, it also will be the path for save figure
        yes: bool, save or not save this figure
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        # plot fundamental figure : SM
        ax1.bar(self.Date, self.SM, label="observation SM", color="cornflowerblue", alpha=0.5)  # sm bar
        ax2.plot(self.Date, self.SM_percentile, label="SM Percentile", color="royalblue",
                 linewidth=0.9)  # sm_percentile
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=0.5), label=f"Mean=0.5",
                 color="sandybrown")  # mean
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=self.threshold),
                 label=f"Threshold={self.threshold}", color="chocolate")  # threshold
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=self.eliminate_threshold),
                 label=f"eliminate_threshold={self.eliminate_threshold}", color="darkred")  # eliminate_threshold
        # plot the trend line of sm
        z = np.polyfit(range(len(self.Date)), self.SM, deg=1)
        p = np.poly1d(z)
        ax1.plot(self.Date, p(range(len(self.Date))), color="brown", alpha=0.5, label=f"Trend:{p}")
        # plot drought events
        events = np.full((len(self.Date),), fill_value=self.threshold)
        for i in range(len(self.dry_flag_start)):
            events[self.dry_flag_start[i]:self.dry_flag_end[i] + 1] = \
                self.SM_percentile[self.dry_flag_start[i]:self.dry_flag_end[i] + 1]
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]
            ax2.plot(self.Date[start:end + 1], self.SM_percentile[start:end + 1], "r", linewidth=1)
        ax2.fill_between(self.Date, events, self.threshold, alpha=0.8, facecolor="peru",
                         label="Drought events", interpolate=True)
        # plot flash drought (from compute start point: means change from no-drought)
        for i in range(len(self.fd_flag_start)):
            for j in range(len(self.fd_flag_start[i])):
                start = self.fd_flag_start[i][j]
                end = self.fd_flag_end[i][j]
                ax2.plot(self.Date[start:end + 1], self.SM_percentile[start:end + 1], color="purple", linestyle='--',
                         linewidth=1.5, marker=7, markersize=6)
                print(f"drought: {i}", ";", f"flash develop period: {j}")
        # set figure
        ax1.set_ylabel("SM")
        ax2.set_ylabel("SM Percentile")
        ax2.set_xlabel("Date")
        ax1.set_xlim(xmin=self.Date[0], xmax=self.Date[-1])
        ax2.set_xlim(xmin=self.Date[0], xmax=self.Date[-1])
        ax1.legend(loc="upper right")
        ax2.legend(loc="upper right")
        ax1.set_title(title, loc="center")
        ax1.set_xticks(self.Date[::int(len(self.Date) / 6)])  # set six ticks
        ax1.set_xticklabels(self.Date_tick[::int(len(self.Date) / 6)])
        ax2.set_xticks(self.Date[::int(len(self.Date) / 6)])  # set six ticks
        ax2.set_xticklabels(self.Date_tick[::int(len(self.Date) / 6)])
        plt.show()
        if yes == 1:
            plt.savefig("Drought/" + title)

    def FD_character_plot(self, yes=0):
        """ plot the boxplot of flash drought characters: RImean/RImax/FDD/FDS"""
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
        if yes == 1:
            plt.savefig("FD_character_boxplot")


if __name__ == "__main__":
    np.random.seed(15)
    sm = np.random.rand(365 * 3, )
    sm = np.convolve(sm, np.repeat(1 / 2, 3), mode='full')  # running means
    tc = 10
    pc = 0.5
    rds = 0.41
    fd_pc = 0.2
    fd_tc = 2
    FD1 = FD(sm, 365, tc=tc, pc=pc, rds=rds, eliminating=True, fd_pooling=True, fd_tc=fd_tc, fd_pc=fd_pc, fd_excluding=False)
    sm_per = FD1.SM_percentile
    RI = FD1.RI
    FD1.plot()
    print(FD1.out_put())
    out = FD1.out_put()
    dp = sum(FD1.dp)

    # D1 = Drought(sm, 365, tc=tc, pc=pc, rds=rds)
    # sm_per = D1.SM_percentile
    # D1.plot()
    # print(D1.out_put())
    # out1 = D1.out_put()
    # D2 = Drought(sm, 365, pooling=False, excluding=False, tc=tc, pc=pc, rds=rds)
    # D2.plot()
    # print(D2.out_put())
    # out2 = D2.out_put()
    # D3 = Drought(sm, 365, pooling=True, excluding=False, tc=tc, pc=pc, rds=rds)
    # D3.plot()
    # print(D3.out_put())
    # out3 = D3.out_put()
    # D4 = Drought(sm, 365, pooling=False, excluding=True, tc=tc, pc=pc, rds=rds)
    # D4.plot()
    # print(D4.out_put())
    # out4 = D4.out_put()
    # np.savetxt("1.txt", D1.SM_percentile)
    # out.to_excel("1.xlsx")
    # out2.to_excel("2.xlsx")
    # out3.to_excel("3.xlsx")
    # out4.to_excel("4.xlsx")
