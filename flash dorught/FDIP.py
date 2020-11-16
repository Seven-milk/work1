# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Flash Drought Identify Process
# reference:
# [1] Two Different Methods for Flash Drought Identification: Comparison of Their Strengths and Limitations
# [2] A new framework for tracking flash drought events in space and time
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
    def percentile(x: np.ndarray, path="1", y=0, bw_method="scott") -> np.ndarray:
        """ calculate the percentile for each point in x, using kernel density estimation, bw_method='Scott'
        input:
            x: 1D numpy.ndarray
        output:
            x_percentile: 1D numpy.ndarray
        """
        kde = stats.gaussian_kde(x, bw_method=bw_method)
        x_percentile = np.array([kde.integrate_box_1d(low=0, high=x[i]) for i in range(len(x))])
        # plot while y set to 1
        if y == 1:
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.hist(x, bins=int(len(x)*kde.covariance_factor()), label="Hist", alpha=0.5)
            x_eval = np.linspace(x.min(), x.max(), num=(x.max() - x.min()) * 100)
            ax1.plot(x, np.zeros(x.shape), '+', color='navy', ms=20, label="Samples")
            ax1.set_ylabel("Number of samples")
            ax2.plot(x_eval, kde(x_eval), 'r-', label="KDE based on bw_method: " + bw_method)
            ax2.set_ylabel("PDF")
            ax1.set_title("Kernel density estimation")
            ax1.set_xlim(x.min(), x.max())
            plt.legend([ax1, ax2])
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
            plt.savefig(f"/SM_kde/SM_kde{path}")
        return x_percentile

    def cal_SM_percentile(self) -> np.ndarray:
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
            # l百分比计算
            l = self.percentile(l)
            SM_percentile[:len(l), i] = l
        SM_percentile = SM_percentile.flatten()
        SM_percentile = SM_percentile[~np.isnan(SM_percentile)]
        return SM_percentile


class Drought(SM_percentile):
    def __init__(self, SM, timestep=365, Date=0, threshold1=0.4, threshold2=0.2):
        """
        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold1: the threshold to identify dry bell
            threshold2: the threshold to eliminate mild drought events
            Date: the Date of SM, np.ndarray
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
        if Date == 0:
            self.Date = list(range(len(SM)))
        else:
            self.Date = Date
        self.timestep = timestep
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        SM_percentile.__init__(self, SM, timestep)
        self.dry_flag_start, self.dry_flag_end = self.run_threshold(self.SM_percentile, self.threshold1)
        self.eliminate()
        self.DD, self.DS, self.SM_min, self.SM_min_flag = self.character()

    @staticmethod
    def run_threshold(index: np.ndarray, threshold: float) -> (np.ndarray, np.ndarray):
        """ run_threshold to identify dry bell (start-end)
        index: the base index
        threshold: the threshold to identify dry bell(index < threshold)
        point explain(discrete): start < threshold, end < threshold --> it is shrinkable and strict
        """
        # define drought based on index and threshold
        dry_flag = np.argwhere(index <= threshold).flatten()
        dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()[1:]
        dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()[:-1]
        if index[dry_flag[0]] <= threshold:
            dry_flag_start = np.insert(dry_flag_start, 0, dry_flag[0])
        if index[dry_flag[-1]] <= threshold:
            dry_flag_end = np.append(dry_flag_end, dry_flag[-1])
        return dry_flag_start, dry_flag_end

    def eliminate(self):
        """ eliminate mild drought events which are not dry based on threshold2 """
        index = []
        for i in range(len(self.dry_flag_start)):
            if min(self.SM_percentile[self.dry_flag_start[i]:self.dry_flag_end[i] + 1]) > self.threshold2:
                index.append(i)
        self.dry_flag_start, self.dry_flag_end = np.delete(self.dry_flag_start, index), np.delete(self.dry_flag_end,
                                                                                                  index)
        # eliminate mild drought with short duration or small severity

    def character(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """ extract the drought character variables
        DD: duration of drought events (unit based on timestep)
        DS: severity of drought events (based on SM_percentile)
        SM_min: the min value of SM (based on SM_percentile)
        """
        n = len(self.dry_flag_start)  # the number of flash drought events
        DD = self.dry_flag_end - self.dry_flag_start + 1
        x = self.threshold1 - self.SM_percentile
        DS = np.array([x[self.dry_flag_start[i]: self.dry_flag_end[i] + 1].sum() for i in range(n)], dtype='float')
        SM_min = np.array([min(self.SM_percentile[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) for i in range(n)]
                          , dtype='float')
        SM_min_flag = np.array([np.argmin(self.SM_percentile[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) +
                                self.dry_flag_start[i] for i in range(n)], dtype='float')
        return DD, DS, SM_min, SM_min_flag

    def out_put(self, xlsx=0) -> pd.DataFrame:
        """ output the drought event to a dataframe and .xlsx file(to set xlsx=1) """
        n = len(self.dry_flag_start)  # the number of flash drought events
        Date_start = np.array([self.Date[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date[self.dry_flag_end[i]] for i in range(n)])
        threshold1 = np.full((n,), self.threshold1, dtype='float')
        threshold2 = np.full((n,), self.threshold2, dtype='float')
        Drought_character = pd.DataFrame(
            np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.SM_min,
                       self.SM_min_flag, threshold1, threshold2)).T,
            columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "SM_min", "SM_min_flag"
                     , "thrshold1", "threshold2"))
        if xlsx == 1:
            Drought_character.to_excel("/Drought_character", index=False)
        return Drought_character

    def plot(self, title="Drought", yes=0):
        """ plot the drought events: time series of index; threshold1/2; drought events
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
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=self.threshold1),
                 label=f"Threshold1={self.threshold1}", color="chocolate")  # threshold1
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=self.threshold2),
                 label=f"Threshold2={self.threshold2}", color="darkred")  # threshold2
        # plot the trend line of sm
        z = np.polyfit(range(len(self.Date)), self.SM, deg=1)
        p = np.poly1d(z)
        ax1.plot(self.Date, p(range(len(self.Date))), color="brown", alpha=0.5, label=f"Trend:{p}")
        # plot drought events
        events = np.full((len(self.Date),), fill_value=self.threshold1)
        for i in range(len(self.dry_flag_start)):
            events[self.dry_flag_start[i]:self.dry_flag_end[i] + 1] = \
                self.SM_percentile[self.dry_flag_start[i]:self.dry_flag_end[i] + 1]
        ax2.fill_between(self.Date, events, self.threshold1, alpha=0.5, facecolor="red",
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
        plt.show()
        if yes == 1:
            plt.savefig("Drought/" + title)


class FD(Drought):
    def __init__(self, SM, timestep=365, Date=0, threshold1=0.4, threshold2=0.2, RI_threshold=0.05):
        """
        Identify flash drought based on rule: RI > threshold -> fd_flag_start, fd_flag_end
        elimate...

        this identification regard flash drought as a flash develop period of a normal drought event, and a normal
        drought can contain more than one flash drought

        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold1: the threshold to identify dry bell
            threshold2: the threshold to eliminate mild drought events
            RI_threshold: the threshold of RI to extract extract flash development period(RI instantaneous)
            Date: the Date of SM, np.ndarray
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
        Drought.__init__(self, SM, timestep, Date, threshold1, threshold2)
        self.RI_threshold = RI_threshold
        self.fd_flag_start, self.fd_flag_end, self.RImean, self.RImax, self.RI = self.develop_period()
        self.fd_eliminate()
        self.dp, self.FDD, self.FDS = self.fd_character()

    @staticmethod
    def fd_run_threshold(index, threshold):
        """ run_threshold to identify develop period (index > threshold, different with run_threshold)
        index: the base index
        threshold: the threshold to identify develop period(index > threshold)
        point explain(discrete): start > threshold, end > threshold --> it is shrinkable and strict
        """
        # define develop period based on index and threshold
        dry_flag = np.argwhere(index >= threshold).flatten()
        dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()[1:]
        dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()[:-1]
        # because the rare of develop period(index(RI) > threshold) when the threshold is not zero, dry_flag may be a []
        # , a check is nesscessary
        # ps: if threshold set as zero, it cant be [], because of a drought event must have +- RI, dry_flag cant be a []
        if len(dry_flag) > 0:
            if index[dry_flag[0]] >= threshold:
                dry_flag_start = np.insert(dry_flag_start, 0, dry_flag[0])
            if index[dry_flag[-1]] >= threshold:
                dry_flag_end = np.append(dry_flag_end, dry_flag[-1])
        return dry_flag_start, dry_flag_end

    def develop_period(self) -> (list, list, list, list):
        """ extract extract flash development period of drought event, RI is instantaneous
        unit = 1/time interval of input SM

        return fd_flag_start, fd_flag_end, RImean, RImax for each flash drought event of every drought events
        list[array, array, ...] --> list: n(drought) array: m(flash)
        SM[--drought1[--flash develop period1[--fd_flag_start, fd_flag_end, RImean, RImax--],..flash develop periodn[]--], ...droughtn[]--]
        """
        n = len(self.dry_flag_start)  # the number of drought events
        #  distinguish the compute start point and flag start point:
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
            fd_flag_start.append(fd_flag_start_)  # np.ndarray
            fd_flag_end.append(fd_flag_end_)  # np.ndarray
            RImean.append(RImean_)  # np.ndarray
            RImax.append(RImax_)
        return fd_flag_start, fd_flag_end, RImean, RImax, RI

    def fd_eliminate(self):
        """ eliminate develop period which are not flash based on RI_mean_threshold of each drought event;
        However, this definition based on RI has ensured large enough RI, so dont have to elimate based on RI_mean
        """
        """
        n = len(self.dry_flag_start)  # the number of drought events
        indexi = []
        indexj = []
        for i in range(n):
            m = len(self.fd_flag_start[i])  # the number of flash develop period of each drought event
            for j in range(m):
                if self.RImean[i][j] < self.RI_mean_threshold:
                    indexi.append(i)
                    indexj.append(j)
        if len(indexi) > 0:
            for k in range(len(indexi)):
                self.fd_flag_start[indexi[k]] = np.delete(self.fd_flag_start[indexi[k]], indexj[k])
                self.fd_flag_end[indexi[k]] = np.delete(self.fd_flag_end[indexi[k]], indexj[k])
                self.RImean[indexi[k]] = np.delete(self.RImean[indexi[k]], indexj[k])
                self.RImax[indexi[k]] = np.delete(self.RImax[indexi[k]], indexj[k])
        """
        # TODO add rule to elominate drought event with RI_mean > RImean_threshold but duration too short
        #  合并连续骤旱（合并发展阶段），剔除小骤旱

    def fd_character(self) -> (list, list, list):
        """ extract the drought character variables
        the develop period number of each drought
        FDD: duration of a flash drought event(develop period)
        FDS: severity of a flash drought event
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
                x = self.threshold1 - self.SM_percentile  # np.ndarray
                FDS_.append(x[start: end + 1].sum())
            FDD.append(FDD_)
            FDS.append(FDS_)
        return dp, FDD, FDS

    def out_put(self, xlsx=0) -> pd.DataFrame:
        """ output the drought event to a dataframe and .xlsx file(to set xlsx=1) """
        n = len(self.dry_flag_start)  # the number of flash drought events
        Date_start = np.array([self.Date[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date[self.dry_flag_end[i]] for i in range(n)])
        threshold1 = np.full((n,), self.threshold1, dtype='float')
        threshold2 = np.full((n,), self.threshold2, dtype='float')
        RI_threshold = np.full((n,), self.RI_threshold, dtype='float')
        Drought_character = pd.DataFrame(
            np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.SM_min,
                       self.SM_min_flag, threshold1, threshold2, RI_threshold)).T,
            columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "SM_min", "SM_min_flag",
                     "thrshold1", "threshold2", "RI_threshold"))
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
        """ plot the drought events: time series of index; threshold1/2; drought events
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
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=self.threshold1),
                 label=f"Threshold1={self.threshold1}", color="chocolate")  # threshold1
        ax2.plot(self.Date, np.full((len(self.Date),), fill_value=self.threshold2),
                 label=f"Threshold2={self.threshold2}", color="darkred")  # threshold2
        # plot the trend line of sm
        z = np.polyfit(range(len(self.Date)), self.SM, deg=1)
        p = np.poly1d(z)
        ax1.plot(self.Date, p(range(len(self.Date))), color="brown", alpha=0.5, label=f"Trend:{p}")
        # plot drought events
        events = np.full((len(self.Date),), fill_value=self.threshold1)
        for i in range(len(self.dry_flag_start)):
            events[self.dry_flag_start[i]:self.dry_flag_end[i] + 1] = \
                self.SM_percentile[self.dry_flag_start[i]:self.dry_flag_end[i] + 1]
        ax2.fill_between(self.Date, events, self.threshold1, alpha=0.5, facecolor="red",
                         label="Drought events", interpolate=True)
        # plot flash drought (from compute start point: means change from no-drought)
        for i in range(len(self.fd_flag_start)):
            for j in range(len(self.fd_flag_start[i])):
                if self.fd_flag_start[i][j] == 0:  # check if it is the first point
                    start = self.fd_flag_start[i][j]
                else:
                    start = self.fd_flag_start[i][j] - 1
                end = self.fd_flag_end[i][j]
                ax2.plot(self.Date[start:end + 1], self.SM_percentile[start:end + 1], color="r", linewidth=2, marker=7,
                         markersize=6)
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
        plt.show()
        if yes == 1:
            plt.savefig("Drought/" + title)


class FD_RI(FD):
    def __init__(self, SM, timestep=365, Date=0, threshold1=0.4, threshold2=0.2, RI_threshold=0.1):
        """
        Identify flash drought based on rule: RI > threshold -> fd_flag_end
        But fd_flag_start = dry_flag_start and there is no eliminate procedure for not flash event
        (this identification class inherit FD and modify develop_period function and fd_eliminate function)

        each dorught have no more than one flash drought development

        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold1: the threshold to identify dry bell
            threshold2: the threshold to eliminate mild drought events
            RI_threshold: the threshold of RI to eliminate mild flash drought events which not flash(RI max)
            Date: the Date of SM
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
        FD.__init__(self, SM, timestep, Date, threshold1, threshold2, RI_threshold)

    def develop_period(self) -> (list, list, list, list):
        """ extract extract flash development period of drought event, RI is instantaneous
        unit = 1/time interval of input SM
        return fd_flag_start, fd_flag_end, RImean (list: keep the same output with FD to use inherited function,
        such as plot and ouput)
        """
        n = len(self.dry_flag_start)  # the number of drought events
        RI = self.SM_percentile - np.append(self.SM_percentile[0], self.SM_percentile[:-1])
        RI = RI * -1
        # calculate RI of the whole timeseries of SM_peritile, then extract drought/flash develop period from it,
        # hypothesize RI in the first position is zero(self.SM_percentile[0]-self.SM_percentile[0])
        # diff= sm[t]-sm[t-1] < 0: develop: RI > 0 ——> multiply factor: -1

        # extract drought/flash develop period from each dorught using RI
        # list[array, array, ...] --> list: n(drought) array: m(flash: 0 or 1)
        fd_flag_start = [np.array([self.dry_flag_start[i]]) for i in range(n)]
        fd_flag_end, RImean, RImax = [], [], []
        for i in range(n):
            start = self.dry_flag_start[i]
            end = self.dry_flag_end[i]
            #  distinguish the compute start point and flag start point:
            # compute start point: compute from -1, it means the develop period from no-drought to drought
            # flag start point: it means the point has been achieved the threshold condition (match the shrinkable
            # run threshold)
            # compute RI_mean from start - 1(compute start point, RI_[0] = SM[0] - SM[-1]) -> flag start point
            RI_ = RI[start: end + 1]
            RImean_ = np.array([RI_[:i + 1].mean() for i in range(end - start + 1)])
            fd_flag_end_ = np.argwhere(RI_ <= self.RI_threshold)
            # check the fd_flag_end_ to see if it is empty
            if len(fd_flag_end_) != 0:
                if fd_flag_end_.flatten()[0] != 0:
                    fd_flag_end_ = fd_flag_end_.flatten()[0]  # index of end based on RI_
                    fd_flag_end_ = fd_flag_end_ - 1  # shrinkable(point <= self.RI_threshold -> point - 1): must be (end
                    # > threshold) -> flag end
                    fd_flag_end.append(np.array([fd_flag_end_ + start]))  # index of end based on SM index/RI,
                    RImean.append(np.array([RImean_[fd_flag_end_]]))
                    RImax.append(np.array([RI_[: fd_flag_end_ + 1].max()]))
                else:
                    fd_flag_end.append(np.array([]))  # There is no flash drought
                    fd_flag_start[i] = np.array([])
                    RImean.append(np.array([]))
                    RImax.append(np.array([]))
            else:
                fd_flag_end.append(np.array([]))  # There is no flash drought
                fd_flag_start[i] = np.array([])
                RImean.append(np.array([]))
                RImax.append(np.array([]))
        return fd_flag_start, fd_flag_end, RImean, RImax, RI

    def fd_eliminate(self):
        """ There is no eliminate procedure for not flash event """
        pass


if __name__ == "__main__":
    sm = np.random.rand(365 * 5, )
    sm = np.convolve(sm, np.repeat(1 / 3, 3), mode='full')  # running means
    FD1 = FD_RI(sm, 365)
    RI = FD1.RI
    FD1.plot()
    print(FD1.out_put())
    out = FD1.out_put()
