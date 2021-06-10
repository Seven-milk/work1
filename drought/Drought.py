# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Drought Identify

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import draw_plot
import useful_func


class Drought:
    def __init__(self, drought_index, Date_tick, threshold=0.4, pooling=True, tc=1, pc=0.2, excluding=True,
                 rds=0.41):
        """
        input:
            drought_index: 1D list or numpy array, drought index
            threshold: the threshold to identify dry bell
            Date_tick: the Date of drought index for plotting, np.ndarray, Date_tick equal to list(range(len(SM))) while
                        Date_tick=[](default)

            pooling: bool, whether activate pooling: pooing while (ti < tc) & (vi/si < pc), based on IC method
                tc: predefined critical duration
                pc: pooling ratio, the critical ratio of excess volume(vi) of inter-event time and the preceding deficit
                    volume(si)
            excluding: bool, whether activate excluding: excluding while (rd = di / dmean < rds) and (rs = si / smean <
                    rds), based on IC method
                rds: predefined critical excluding ratio, compare with rd/rs = d/s / d/s_mean

        output:
            self.dry_flag_start, self.dry_flag_end: start end of each drought events(shrinkable)
            self.DD, self.DS, self.SM_min: character of each drought events
            plot: use self.plot, save figure by setting save_on = True
            out_put: use self.out_put(self, xlsx=1) , set xlsx=1: out put drought and save as xlsx

        """
        # input set
        if type(drought_index) is np.ndarray:
            self.drought_index = drought_index
        else:
            self._index = np.array(drought_index, dtype='float')

        if len(Date_tick) == 0:
            self.Date_tick = list(range(len(drought_index)))
        else:
            if type(drought_index) is np.ndarray:
                self.Date_tick = Date_tick
            else:
                self.Date_tick = np.array(Date_tick, dtype='float')

        # general set
        self.Date = list(range(len(drought_index)))
        self.threshold = threshold
        self.pooling = pooling
        self.tc = tc
        self.pc = pc
        self.excluding = excluding
        self.rds = rds

        # drought identify
        self.dry_flag_start, self.dry_flag_end = self.run_threshold(self.drought_index, self.threshold)

        # pooling
        if self.pooling:
            self.cal_pooling()

        # character
        self.DD, self.DS, self.index_min, self.index_min_flag = self.character()

        # excluding, should be after character
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
        """ pooling while (ti < tc) & (vi/si < pc), based on IC method """
        size = len(self.dry_flag_start)
        i = 0
        while i < size - 1:
            ti = self.dry_flag_start[i + 1] - self.dry_flag_end[i]
            vi = (self.drought_index[self.dry_flag_end[i] + 1: self.dry_flag_start[i + 1]] - self.threshold).sum()
            si = (self.threshold - self.drought_index[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]).sum()
            if (ti <= self.tc) and ((vi / si) <= self.pc):
                self.dry_flag_end[i] = self.dry_flag_end[i + 1]
                self.dry_flag_start = np.delete(self.dry_flag_start, i + 1)
                self.dry_flag_end = np.delete(self.dry_flag_end, i + 1)
                size -= 1
            else:
                i += 1

    def character(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """ extract the drought character variables

            DD: duration of drought events (unit based on timestep)
            DS: severity of drought events (unit based on drought index)
            index_min/_flag: the min value of index (unit based on index)

        """
        # the number of flash drought events
        n = len(self.dry_flag_start)

        # extract drought character
        DD = self.dry_flag_end - self.dry_flag_start + 1

        # end + 1: to contain the end point, it satisfy drought_index[end] < threshold
        x = self.threshold - self.drought_index
        DS = np.array([x[self.dry_flag_start[i]: self.dry_flag_end[i] + 1].sum() for i in range(n)], dtype='float')

        index_min = np.array([min(self.drought_index[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) for i in range(n)],
                          dtype='float')

        # add self.dry_flag_start to revert flag index to make it based on index of drought index
        index_min_flag = np.array([np.argmin(self.drought_index[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) +
                                self.dry_flag_start[i] for i in range(n)], dtype='float')

        return DD, DS, index_min, index_min_flag

    def cal_excluding(self):
        """ excluding while (rd = di / dmean < rds) and (rs = si / smean < rds), based on IC method """
        RD = self.DD / self.DD.mean()
        RS = self.DS / self.DS.mean()
        size = len(RD)
        i = 0
        while i < size:
            if (RD[i] <= self.rds) and (RS[i] <= self.rds):  # note: change or to and, avoid to excluding short serious
                # events
                self.dry_flag_start = np.delete(self.dry_flag_start, i)
                self.dry_flag_end = np.delete(self.dry_flag_end, i)
                self.DD = np.delete(self.DD, i)
                self.DS = np.delete(self.DS, i)
                self.index_min = np.delete(self.index_min, i)
                self.index_min_flag = np.delete(self.index_min_flag, i)
                RD = np.delete(RD, i)
                RS = np.delete(RS, i)
                size -= 1
            else:
                i += 1

    def out_put(self, xlsx_on=False) -> pd.DataFrame:
        """ output the drought event to a dataframe and .xlsx file(to set xlsx_on=True) """
        # the number of flash drought events
        n = len(self.dry_flag_start)

        # calculate other characters which will be put into output
        Date_start = np.array([self.Date_tick[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date_tick[self.dry_flag_end[i]] for i in range(n)])
        threshold = np.full((n,), self.threshold, dtype='float')

        # build output character
        Drought_character = pd.DataFrame(
            np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.index_min,
                       self.index_min_flag, threshold)).T,
            columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "index_min", "index_min_flag",
                     "threshold"))

        # save
        if xlsx_on == True:
            Drought_character.to_excel("Drought_character.xlsx", index=False)

        return Drought_character

    def plot(self, title="Drought", save_on=False):
        """ plot the drought events: time series of index; threshold; drought events
        title: string, the title of this figure, it also will be the path for save figure
        save_on: bool, save or not save this figure
        """
        # fig
        fig = draw_plot.Figure()
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="Drought Plot", labelx="Date", labely="Drought Index",
                              legend_on=True, xlim=(self.Date[0], self.Date[-1]))

        # drought index series
        drought_index_plot = draw_plot.PlotDraw(self.Date, self.drought_index, label="drought index",
                                                color="cornflowerblue", alpha=0.5, linewidth=0.5, zorder=30)
        draw.adddraw(drought_index_plot)

        # threshold series
        threshold_series = np.full((len(self.Date),), fill_value=self.threshold)
        threshold_plot = draw_plot.PlotDraw(self.Date, threshold_series, label=f"Threshold: {self.threshold}",
                                            color="chocolate", linewidth=1)
        draw.adddraw(threshold_plot)

        # trend series
        z = np.polyfit(range(len(self.Date)), self.drought_index, deg=1)
        p = np.poly1d(z)
        trend_series = p(range(len(self.Date)))
        trend_plot = draw_plot.PlotDraw(self.Date, trend_series, color="brown", alpha=0.5, label=f"Trend line")
        draw.adddraw(trend_plot)

        # drought events
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
            # plot line
            # start = self.dry_flag_start[i]
            # end = self.dry_flag_end[i]
            # events_plot = draw_plot.PlotDraw(self.Date[start:end + 1], self.drought_index[start:end + 1], "r",
            # linewidth=1)
            # draw.adddraw(events_plot)

        # set ticks and labels
        fig.ax.set_xticks(self.Date[::int(len(self.Date) / 6)])  # set six ticks
        fig.ax.set_xticklabels(self.Date_tick[::int(len(self.Date) / 6)])
        fig.show()

        # save
        if save_on == True:
            plt.savefig("Drought_" + title + ".tiff")

    def Drought_character_plot(self, yes=0):
        """ plot the boxplot of flash drought characters: RImean/RImax/FDD/FDS """
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.boxplot(self.DD)  # np.array([i for j in FD1.RImean for i in j]).min()
        plt.title("DD")
        plt.subplot(2, 2, 2)
        plt.boxplot(self.DS)
        plt.title("DS")
        plt.subplot(2, 2, 3)
        plt.boxplot(self.index_min)
        plt.title("index_min")
        plt.subplot(2, 2, 4)
        plt.boxplot(self.drought_index)
        plt.title("drought_index")
        if yes == 1:
            plt.savefig("Drought_character_boxplot")


if __name__ == '__main__':
    # test code through using a random series(with a random seed)
    np.random.seed(15)
    drought_index = np.random.rand(365 * 3, )
    drought_index = np.convolve(drought_index, np.repeat(1 / 2, 3), mode='full')  # running means
    tc = 10
    pc = 0.5
    rds = 0.41
    fd_pc = 0.2
    fd_tc = 2
    d = Drought(drought_index, Date_tick=[], tc=tc, pc=pc, rds=rds)
    d.plot()
    print(d.out_put())

    # ----------------------- compare results with different configuration -----------------------
    # D1
    D1 = Drought(drought_index, Date_tick=[], tc=tc, pc=pc, rds=rds)
    D1.plot()
    out1 = D1.out_put()
    # D2
    D2 = Drought(drought_index, Date_tick=[], pooling=False, excluding=False, tc=tc, pc=pc, rds=rds)
    D2.plot()
    out2 = D2.out_put()
    # D3
    D3 = Drought(drought_index, Date_tick=[], pooling=True, excluding=False, tc=tc, pc=pc, rds=rds)
    D3.plot()
    out3 = D3.out_put()
    # D4
    D4 = Drought(drought_index, Date_tick=[], pooling=False, excluding=True, tc=tc, pc=pc, rds=rds)
    D4.plot()
    out4 = D4.out_put()