# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Flash Drought Identify Process
# reference:Two Different Methods for Flash Drought Identification: Comparison of Their Strengths and Limitations
import numpy as np
import pandas as pd


class Drought():
    def __init__(self, SM, timestep, Date=0, threshold1=0.4, threshold2=0.2):
        """
        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold1: the threshold to identify dry bell
            threshold2: the threshold to eliminate mild drought events
            Date: the Date of SM
            timestep: the timestep of SM （the number of SM data in one year）, which is used to do cal_SM_percentile
            (reshape a vector to a array, which shape is (n(year)+1 * timestep))
                365 : daily, 365 data in one year
                12 : monthly, 12 data in one year
                73 : pentad(5), 73 data in one year
                x ：x data in one year
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
        self.SM_percentile = self.cal_SM_percentile()
        self.dry_flag_start, self.dry_flag_end = self.run_threshold()
        self.eliminate()
        self.DD, self.DS, self.SM_min = self.character()

    @staticmethod
    def percentile(x: np.ndarray) -> np.ndarray:
        """ calculate the percentile for each point in x """
        return x

    def cal_SM_percentile(self) -> np.ndarray:
        """ calculate SM percentile using SM, with process of timestep(e.g. daily) """
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

    def run_threshold(self) -> (np.ndarray, np.ndarray):
        """ run_threshold to identify dry bell """
        # define drought based on threshold1
        dry_flag = np.argwhere(self.SM_percentile <= self.threshold1).flatten()
        dry_flag_start = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, 1).flatten() != 1)].flatten()[1:]
        dry_flag_end = dry_flag[np.argwhere(dry_flag - np.roll(dry_flag, -1).flatten() != -1)].flatten()[:-1]
        if self.SM_percentile[dry_flag[0]] <= self.threshold1:
            dry_flag_start = np.insert(dry_flag_start, 0, dry_flag[0])
        if self.SM_percentile[dry_flag[-1]] <= self.threshold1:
            dry_flag_end = np.append(dry_flag_end, dry_flag[-1])
        return dry_flag_start, dry_flag_end

    def eliminate(self):
        """ eliminate mild drought events which are not dry based on threshold2 """
        index = []
        for i in range(len(self.dry_flag_start)):
            if min(self.SM_percentile[self.dry_flag_start[i]:self.dry_flag_end[i] + 1]) > self.threshold2:
                index.append(i)
        self.dry_flag_start, self.dry_flag_end = np.delete(self.dry_flag_start, index), np.delete(self.dry_flag_end, index)
        # TODO eliminate mild drought with short duration or small severity

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
        return DD, DS, SM_min

    def out_put(self, xlsx=0) -> pd.DataFrame:
        """output the drought event to a dataframe and .xlsx file(to set xlsx=1)"""
        n = len(self.dry_flag_start)  # the number of flash drought events
        Date_start = np.array([self.Date[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date[self.dry_flag_end[i]] for i in range(n)])
        Drought_character = pd.DataFrame(np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.SM_min)).T,
                                         columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "SM_min"))
        if xlsx == 1:
            Drought_character.to_excel("/Drought_character", index=False)
        return Drought_character


class FD(Drought):
    def __init__(self, SM, timestep, Date=0, threshold1=0.4, threshold2=0.2, RImean_threshold=0.065,
                 RImax_threshold=0.1):
        """
        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold1: the threshold to identify dry bell
            threshold2: the threshold to eliminate mild drought events
            RImean_threshold: the threshold of RImean to identify flash drought
            RImax_threshold: the threshold of RImax to identify flash drought
            Date: the Date of SM
            timestep: the timestep of SM （the number of SM data in one year）, which is used to do cal_SM_percentile
            (reshape a vector to a array, which shape is (n(year)+1 * timestep))
                365 : daily, 365 data in one year
                12 : monthly, 12 data in one year
                73 : pentad(5), 73 data in one year
                x ：x data in one year
        """
        Drought.__init__(self)
        self.RImean_threshold = RImean_threshold
        self.RImax_threshold = RImax_threshold
        self.dry_flag_end_fd, self.RI_mean, self.RI_max= self.cal_RI()

    # def cal_RI(self) -> (np.ndarray, np.ndarray, np.ndarray):
    #     """ extract extract flash development period of drought event
    #      and calculate RImean and RImax for each flash drought event, unit = 1/time interval of input SM """
    #     n = len(self.dry_flag_start)  # the number of flash drought events
    #     for i in range(n):
    #         if i==0 & self.dry_flag_start[0] == 0:
    #             RI = self.SM_percentile[]
    #         else:
    #             start = self.dry_flag_start[i]
    #             end = self.dry_flag_end[i]
    #             RI = self.SM_percentile[start, end + 1] - self.SM_percentile[start-1, end]
    #             flag_end_fd_index = RI[]
    #     RI_mean = np.full((n,), np.NAN, dtype='float')
    #     RI_max = np.full((n,), np.NAN, dtype='float')
    #     for i in range(n):
    #         if i == 0 & self.dry_flag_start[0] == 0:
    #             start = self.dry_flag_start[0]
    #             end = self.dry_flag_end[0]
    #             m = self.dry_flag_end[i] - self.dry_flag_start[
    #                 i] + 1  # the number of drought duration of each event
    #             RI = [0].extend(
    #                 [self.SM_percentile[j] - self.SM_percentile[j - 1] for j in range(start + 1, end + 1)])
    #             RI_mean[0] = sum(RI) / m
    #             RI_max[0] = max(RI)
    #         else:
    #             start = self.dry_flag_start[i]
    #             end = self.dry_flag_end[i]
    #             m = self.dry_flag_end[i] - self.dry_flag_start[i] + 1
    #             RI = [self.SM_percentile[j] - self.SM_percentile[j - 1] for j in range(start, end + 1)]
    #             RI_mean[i] = sum(RI) / m
    #             RI_max[i] = max(RI)
    #     return RI_mean, RI_max
        # #
        # n = len(dry_flag_start)
        # for i in range(n):
        #     if i == 0 & dry_flag_start[0] == 0:
        #         start = dry_flag_start[0]
        #         end = dry_flag_end[0]
        #         RI = [0]
        #         for j in range(start + 1, end + 1):
        #             RI.append(self.SM_percentile[j] - self.SM_percentile[j - 1])
        #         dry_flag_end_fd = [for k in range()]
        #
        #
        #         # cal RImean RImax
        #         m = dry_flag_end[i] - dry_flag_start[i] + 1  # the number of drought duration of each event
        #
        #
        #     else:
        #         start = self.dry_flag_start[i]
        #         end = self.dry_flag_end[i]
        #         m = self.dry_flag_end[i] - self.dry_flag_start[i] + 1
        #         RI = [self.SM_percentile[j] - self.SM_percentile[j - 1] for j in range(start, end + 1)]

        # dry_flag_end_fd

    # def eliminate(self):
    #     """ eliminate mild drought events which are not dry based on threshold2
    #     and mild drought events which are not flash based on RImean_threshold / self.RImean_threshold"""
    #     for i in range(len(self.dry_flag_start)):
    #         if min(self.SM_percentile[self.dry_flag_start[i]:self.dry_flag_end[i]]) > self.threshold2 \
    #                 | self.RI_mean[i] <= self.RImean_threshold | self.RI_max <= self.RImean_threshold:
    #             del self.dry_flag_start[i], self.dry_flag_end[i]
    #     # TODO add rule to elominate drought event with RI_mean > RImean_threshold and RI_max > RImax_threshold
    #     #  but duration too short
    #
    # def character(self) -> pd.DataFrame:
    #     """ extract the drought character variables
    #     RI_mean/RI_max
    #     FDD: duration of a flash drought event
    #     FDS: severity of a flash drought event
    #     """
    #     FD_character = pd.DataFrame(columns=("Date_start", "Date_end", "flag_start", "flag_end", "RI_mean", "RI_max",
    #                                          "FDD", "FDS", ""))
    #
    #     return FD_character

if __name__ == "__main__":
    sm = np.random.rand(365, )
    D1 = Drought(sm, 365)