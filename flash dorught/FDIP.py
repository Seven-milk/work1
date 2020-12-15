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
        if self.pooling == True:
            self.cal_pooling()
        self.DD, self.DS, self.SM_min, self.SM_min_flag = self.character()
        if self.excluding == True:
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
            ti = self.dry_flag_start[i+1] - self.dry_flag_end[i]
            vi = (self.SM_percentile[self.dry_flag_end[i] + 1: self.dry_flag_start[i+1]]-self.threshold).sum()
            si = (self.threshold - self.SM_percentile[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]).sum()
            if (ti < self.tc) and ((vi / si) < self.pc):
                self.dry_flag_end[i] = self.dry_flag_end[i+1]
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


class FD(Drought):
    def __init__(self, SM, timestep=365, Date_tick=[], threshold=0.4, pooling=True, tc=1, pc=0.2, excluding=True,
                 rds=0.41, RI_threshold=0.05, eliminate_threshold=0.2):
        """
        Identify flash drought based on rule: extract from a drought event(start-1 : end)
        (ps: start/i - 1 means it can represent the rapid change from wet to drought)
            flash intensification : instantaneous RI[i] > RI_threshold -> fd_flag_start = i - 1, fd_flag_end
            drought : eliminate flash drought whose minimal SM_percentile > eliminate_threshold

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
        self.eliminate_threshold = eliminate_threshold
        self.fd_flag_start, self.fd_flag_end, self.RImean, self.RImax, self.RI = self.develop_period()
        self.fd_eliminate()
        self.dp, self.FDD, self.FDS = self.fd_character()

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

            RI = self.SM_percentile - np.append(self.SM_percentile[0], self.SM_percentile[:-1]) * -1

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
        fd_flag_start, fd_flag_end, RImean, RImax = [], [], [], []
        # list, each element is a df_flag_start/end/RImean/RImax series of a drought event
        # namely, it represents the develop periods(number > 1) of a dought event

        # extract drought/flash develop period from each dorught using RI
        for i in range(n):
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]
            RI_ = RI[start: end + 1]
            fd_flag_start_, fd_flag_end_ = self.fd_run_threshold(RI_, self.RI_threshold)  # this flag based on RI_ index
            RImean_ = np.array([RI_[fd_flag_start_[j]: fd_flag_end_[j] + 1].mean() for j in range(len(fd_flag_start_))],
                               dtype="float")
            RImax_ = np.array([RI_[fd_flag_start_[j]: fd_flag_end_[j] + 1].max() for j in range(len(fd_flag_start_))],
                              dtype="float")
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
            RImean.append(RImean_)  # list(np.ndarray)
            RImax.append(RImax_)
        return fd_flag_start, fd_flag_end, RImean, RImax, RI

    def fd_eliminate(self):
        """ To satisfy the drought condition in the flash drought identification
        method : eliminate flash drought whose minimal SM_percentile > eliminate_threshold
        """
        for i in range(len(self.fd_flag_start)):
            size = len(self.fd_flag_start[i])
            j = 0
            while j < size:
                if min(self.SM_percentile[self.fd_flag_start[i][j]: self.fd_flag_end[i][j] + 1]) > self.eliminate_threshold:
                    self.fd_flag_start[i] = np.delete(self.fd_flag_start[i], j)
                    self.fd_flag_end[i] = np.delete(self.fd_flag_end[i], j)
                    self.RImean[i] = np.delete(self.RImean[i], j)
                    self.RImax[i] = np.delete(self.RImax[i], j)
                    size -= 1
                else:
                    j += 1
        # TODO 合并连续骤旱（合并发展阶段），剔除小骤旱
        #  add rule to elominate drought event with RI_mean > RImean_threshold but duration too short

    def fd_character(self) -> (list, list, list):
        """ extract the drought character variables
        the develop period number of each drought
        FDD: duration of a flash drought event(develop period)
        FDS: severity of a flash drought event, equaling to the change during flash drought period
        """
        n = len(self.dry_flag_start)  # the number of drought events
        dp = []  # number of develop periods of each drought event
        FDD, FDS = [], []
        for i in range(n):
            m = len(self.fd_flag_start[i])  # number of flash develop periods of each drought event
            dp.append(m)
            FDD_ = []
            FDS_ = []
            for j in range(m):
                start = self.fd_flag_start[i][j]
                end = self.fd_flag_end[i][j]
                FDD_.append(end - start + 1)
                FDS_.append(self.SM_percentile[end] - self.SM_percentile[start])
            FDD.append(FDD_)
            FDS.append(FDS_)
        return dp, FDD, FDS

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

# class FD_RI(FD):
#     def __init__(self, SM, timestep=365, Date_tick=[], threshold=0.4, threshold2=0.2, RI_threshold=0.05):
#         """
#         Identify flash drought based on rule: RI > threshold -> fd_flag_end
#         different with FD: no more than one flash development in a drought event, fd_flag_start = dry_flag_start
#         and there is no eliminate procedure for not flash event
#         (this identification class inherit FD and modify develop_period function and fd_eliminate function)
#
#         each dorught have no more than one flash drought development
#
#         input:
#             SM: SOIL MOISTURE, list or numpy array
#             threshold: the threshold to identify dry bell
#             threshold2: the threshold to eliminate mild drought events
#             RI_threshold: the threshold of RI to eliminate mild flash drought events which not flash(RI max)
#             Date_tick: the Date of SM
#             timestep: the timestep of SM （the number of SM data in one year）, which is used to do cal_SM_percentile
#             (reshape a vector to a array, which shape is (n(year)+1 * timestep))
#                 365 : daily, 365 data in one year
#                 12 : monthly, 12 data in one year
#                 73 : pentad(5), 73 data in one year
#                 x ：x data in one year
#
#         output:
#             self.SM_percentile: SM_percentile calculated by SM using self.cal_SM_percentile
#             self.dry_flag_start, self.dry_flag_end: start end of each drought events
#             self.DD, self.DS, self.SM_min: character of each drought events
#             fd_flag_start, fd_flag_end, RImean, RImax: character of each drought events's flash develop periods, shrinkable start point
#             plot: use self.plot(plot(self, title="Drought", yes=0)), save figure set yes = 1: plot dorught, plot flash drought using compute start point[-1]
#             xlsx: use self.out_put(self, xlsx=1) , set xlsx=1: out put drought the xlsx
#
#         """
#         FD.__init__(self, SM, timestep, Date_tick, threshold, threshold2, RI_threshold)
#
#     def develop_period(self) -> (list, list, list, list):
#         """ extract extract flash development period of drought event, RI is instantaneous
#         unit = 1/time interval of input SM
#         return fd_flag_start, fd_flag_end, RImean (list: keep the same output with FD to use inherited function,
#         such as plot and ouput)
#         """
#         n = len(self.dry_flag_start)  # the number of drought events
#         RI = self.SM_percentile - np.append(self.SM_percentile[0], self.SM_percentile[:-1])
#         RI = RI * -1
#         # calculate RI of the whole timeseries of SM_peritile, then extract drought/flash develop period from it,
#         # hypothesize RI in the first position is zero(self.SM_percentile[0]-self.SM_percentile[0])
#         # diff= sm[t]-sm[t-1] < 0: develop: RI > 0 ——> multiply factor: -1
#
#         # extract drought/flash develop period from each dorught using RI
#         # list[array, array, ...] --> list: n(drought) array: m(flash: 0 or 1)
#         fd_flag_start = [np.array([self.dry_flag_start[i]]) for i in range(n)]
#         fd_flag_end, RImean, RImax = [], [], []
#         for i in range(n):
#             start = self.dry_flag_start[i]
#             end = self.dry_flag_end[i]
#             #  distinguish the compute start point and flag start point:
#             # compute start point: compute from -1, it means the develop period from no-drought to drought
#             # flag start point: it means the point has been achieved the threshold condition (match the shrinkable
#             # run threshold)
#             # compute RI_mean from start - 1(compute start point, RI_[0] = SM[0] - SM[-1]) -> flag start point
#             RI_ = RI[start: end + 1]
#             RImean_ = np.array([RI_[:i + 1].mean() for i in range(end - start + 1)])
#             fd_flag_end_ = np.argwhere(RI_ <= self.RI_threshold)
#             # check the fd_flag_end_ to see if it is empty
#             if len(fd_flag_end_) != 0:
#                 if fd_flag_end_.flatten()[0] != 0:
#                     fd_flag_end_ = fd_flag_end_.flatten()[0]  # index of end based on RI_
#                     fd_flag_end_ = fd_flag_end_ - 1  # shrinkable(point <= self.RI_threshold -> point - 1): must be (end
#                     # > threshold) -> flag end
#                     fd_flag_end.append(np.array([fd_flag_end_ + start]))  # index of end based on SM index/RI,
#                     RImean.append(np.array([RImean_[fd_flag_end_]]))
#                     RImax.append(np.array([RI_[: fd_flag_end_ + 1].max()]))
#                 else:
#                     fd_flag_end.append(np.array([]))  # There is no flash drought
#                     fd_flag_start[i] = np.array([])
#                     RImean.append(np.array([]))
#                     RImax.append(np.array([]))
#             else:
#                 fd_flag_end.append(np.array([]))  # There is no flash drought
#                 fd_flag_start[i] = np.array([])
#                 RImean.append(np.array([]))
#                 RImax.append(np.array([]))
#         return fd_flag_start, fd_flag_end, RImean, RImax, RI
#
#     def fd_eliminate(self):
#         """ There is no eliminate procedure for not flash event """
#         # TODO 最小历时剔除小骤旱
#         pass


if __name__ == "__main__":
    np.random.seed(15)
    sm = np.random.rand(365 * 3, )
    sm = np.convolve(sm, np.repeat(1 / 3, 3), mode='full')  # running means
    FD1 = FD(sm, 365)
    RI = FD1.RI
    FD1.plot()
    print(FD1.out_put())
    out = FD1.out_put()
