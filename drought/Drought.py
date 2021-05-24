# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# drought identify
# reference:
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import draw_plot


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
            excluding: bool, whether activate excluding: excluding while (rd = di / dmean < rds) or (rs = si / smean <
                    rds), based on IC method

            timestep: the timestep of SM （the number of SM data in one year）, which is used to do cal_SM_percentile
            (reshape a vector to a array, which shape is (n(year)+1 * timestep))
                365 : daily, 365 data in one year
                12 : monthly, 12 data in one year
                73 : pentad(5), 73 data in one year
                x ：x data in one year

        output:
            self.dry_flag_start, self.dry_flag_end: start end of each drought events(shrinkable)
            self.DD, self.DS, self.SM_min: character of each drought events
            plot: use self.plot(plot(self, title="Drought", yes=0)), save figure set yes = 1: plot dorught
            xlsx: use self.out_put(self, xlsx=1) , set xlsx=1: out put drought the xlsx

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

        #  excluding, should be after character
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
        n = len(self.dry_flag_start)  # the number of flash drought events
        DD = self.dry_flag_end - self.dry_flag_start + 1
        x = self.threshold - self.drought_index
        DS = np.array([x[self.dry_flag_start[i]: self.dry_flag_end[i] + 1].sum() for i in range(n)], dtype='float')
        index_min = np.array([min(self.drought_index[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) for i in range(n)],
                          dtype='float')
        index_min_flag = np.array([np.argmin(self.drought_index[self.dry_flag_start[i]: self.dry_flag_end[i] + 1]) +
                                self.dry_flag_start[i] for i in range(n)], dtype='float')
        return DD, DS, index_min, index_min_flag

    def cal_excluding(self):
        """ excluding while (rd = di / dmean < rds) or (rs = si / smean < rds), based on IC method """
        RD = self.DD / self.DD.mean()
        RS = self.DS / self.DS.mean()
        size = len(RD)
        i = 0
        while i < size:
            if (RD[i] <= self.rds) or (RS[i] <= self.rds):
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
        """ output the drought event to a dataframe and .xlsx file(to set xlsx=1) """
        n = len(self.dry_flag_start)  # the number of flash drought events
        Date_start = np.array([self.Date_tick[self.dry_flag_start[i]] for i in range(n)])
        Date_end = np.array([self.Date_tick[self.dry_flag_end[i]] for i in range(n)])
        threshold = np.full((n,), self.threshold, dtype='float')
        Drought_character = pd.DataFrame(
            np.vstack((Date_start, Date_end, self.dry_flag_start, self.dry_flag_end, self.DD, self.DS, self.index_min,
                       self.index_min_flag, threshold)).T,
            columns=("Date_start", "Date_end", "flag_start", "flag_end", "DD", "DS", "index_min", "index_min_flag",
                     "threshold"))
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
                                                color="cornflowerblue", alpha=0.5, linewidth=1)
        draw.adddraw(drought_index_plot)

        # threshold series
        threshold_series = np.full((len(self.Date),), fill_value=self.threshold)
        threshold_plot = draw_plot.PlotDraw(self.Date, threshold_series, label=f"Threshold: {self.threshold}",
                                            color="chocolate")
        draw.adddraw(threshold_plot)

        # trend series
        z = np.polyfit(range(len(self.Date)), self.drought_index, deg=1)
        p = np.poly1d(z)
        trend_series = p(range(len(self.Date)))
        trend_plot = draw_plot.PlotDraw(self.Date, trend_series, color="brown", alpha=0.5, label=f"Trend line")
        draw.adddraw(trend_plot)

        # drought events
        events = np.full((len(self.Date),), fill_value=self.threshold)
        for i in range(len(self.dry_flag_start)):
            events[self.dry_flag_start[i]:self.dry_flag_end[i] + 1] = \
                self.drought_index[self.dry_flag_start[i]:self.dry_flag_end[i] + 1]
            # plot line
            # start = self.dry_flag_start[i]
            # end = self.dry_flag_end[i]
            # events_plot = draw_plot.PlotDraw(self.Date[start:end + 1], self.drought_index[start:end + 1], "r",
            # linewidth=1)
            # draw.adddraw(events_plot)

        # fill drought events
        fig.ax.fill_between(self.Date, events, self.threshold, alpha=0.8, facecolor="r", label="Drought events",
                            interpolate=True, zorder=20)  # peru

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
    np.random.seed(15)
    sm = np.random.rand(365 * 3, )
    sm = np.convolve(sm, np.repeat(1 / 2, 3), mode='full')  # running means
    tc = 10
    pc = 0.5
    rds = 0.41
    fd_pc = 0.2
    fd_tc = 2
    d = Drought(sm, Date_tick=[], tc=tc, pc=pc, rds=rds)
    d.plot()
    print(d.out_put())