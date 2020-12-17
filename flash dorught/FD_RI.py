# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# a old method : Flash Drought Identify Process, now we dont use this method, this file aim to back it up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from FDIP import FD


class FD_RI(FD):
    def __init__(self, SM, timestep=365, Date_tick=[], threshold=0.4, threshold2=0.2, RI_threshold=0.05):
        """
        Identify flash drought based on rule: RI > threshold -> fd_flag_end
        different with FD: no more than one flash development in a drought event, fd_flag_start = dry_flag_start
        and there is no eliminate procedure for not flash event
        (this identification class inherit FD and modify develop_period function and fd_eliminate function)

        each dorught have no more than one flash drought development

        input:
            SM: SOIL MOISTURE, list or numpy array
            threshold: the threshold to identify dry bell
            threshold2: the threshold to eliminate mild drought events
            RI_threshold: the threshold of RI to eliminate mild flash drought events which not flash(RI max)
            Date_tick: the Date of SM
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
        FD.__init__(self, SM, timestep, Date_tick, threshold, threshold2, RI_threshold)

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
        # pass
