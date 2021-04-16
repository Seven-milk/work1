# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# variation detect

import numpy as np
import abc
import draw_plot
import math
from scipy.special import betainc
from matplotlib import pyplot as plt
import curve_fit
from scipy import signal


class VDBase(abc.ABC):
    ''' VDBase abstract class '''

    @abc.abstractmethod
    def detect(self):
        ''' define detect abstarct method '''

    @abc.abstractmethod
    def plot(self):
        ''' define plot abstarct method '''


class BGVD(VDBase):
    ''' BGVD, the BG method to detect variation, which depend on the variation between mean of sub-series

        reference: [1] Bernaola-Galván P., Ivanov P.C., Amaral L.A.N. et al. Scale invariance in the nonstationarity of
        human heart rate[J]. Physical review letters, 2001, 87(16): 168105.

    # TODO 剔除不合理断点
    '''

    def __init__(self, data, confidence: float = 0.95, l: int = 25):
        ''' init function
        input:
            data: 1D array, time series
            confidence: statistic confidence, default=0.95
            l: the min length to split, l should equal or greater than 25, default=25

        output:
            self.allRet: detect results, a list contains each split process, it has field "subSplit", "absIndex",
                        "absIndexMax", "relativeIndexMax" "T" "PTmax" "pass_"
            self.passRet: detect results, similar as self.allRet, but it only contain split process where pass_ == True
            self.bp: break point in a split process that has "pass_" == True(it could be split)
        '''
        if len(data) <= 25:
            raise ValueError("Input data should be longer than 25")
        self._data = data
        self.confidence = confidence
        self.l = l
        self.allRet = self.detect()
        self.passRet = [ret for ret in self.allRet if ret["pass_"] == True]
        self.bp = [ret["absIndexMax"] for ret in self.allRet if ret["pass_"] == True]
        self.plot()

    def detect(self):
        ''' Implement VDBase.detect
        how to detect:
            use a generator to recursive call self.split() until there is no sub series could be split
        '''
        # init subSplits and ret
        ret = []
        absIndex = np.arange(len(self._data))
        subSplits = [{"absIndex": absIndex, "subSplit": self._data}]

        # define generator to do split
        generator = self.generateSub(subSplits, confidence=self.confidence, l=self.l)

        # Call the generator to generate the result
        for ret_ in generator:
            ret.append(ret_)

        return ret

    def plot(self, time_ticks=None, labely="Data", **kwargs):
        ''' Implement VDBase.plot
        input:
            time_ticks: dict {"ticks": ticks, "interval": interval}, the ticks of time, namely axis x
                        note: for ticks: len=len(data), for interval: dtype=int
            labely: the labely of the first ax
            **kwargs: keyword args of subplots, keyword args in Figure init function
        '''
        # define sub mean series
        sub_mean = np.zeros((len(self._data),))
        sorted_bp = sorted(self.bp)
        sorted_bp.insert(0, int(0))
        sorted_bp.append(len(self._data) + 1)
        for i in range(len(sorted_bp) - 1):
            slice_ = slice(sorted_bp[i], sorted_bp[i + 1], 1)
            sub_mean[slice_] = np.mean(self._data[slice_])

        # define time
        time = np.arange(len(self._data))

        # plot
        n = len(self.passRet) if len(self.passRet) <= 6 else 6  # define max fig number = 6
        if n > 0:
            # plot data and mean lines
            fig = draw_plot.FigureVert(n + 1, sharex=True, **kwargs)
            draw_data = draw_plot.Draw(fig.ax[0], fig, gridy=True, title="BG Variation Detect", labely=labely,
                                       xlim=[0, time[-1]])
            line_data = draw_plot.PlotDraw(time, self._data, alpha=0.6, color="gray", linewidth=2)
            line_mean = draw_plot.PlotDraw(time, sub_mean, color="k", linewidth=0.6)

            draw_data.adddraw(line_data)
            draw_data.adddraw(line_mean)

            # plot T series for each split which can be split, namely self.passRet
            for i in range(n):
                # extract variable
                absIndexMax_ = self.passRet[i]["absIndexMax"]
                # relativeIndexMax_ =self.passRet[i]["relativeIndexMax"]
                PTmax_ = self.passRet[i]["PTmax"]
                absIndex = self.passRet[i]["absIndex"]
                T_ = np.zeros((len(self._data),))
                T_[absIndex] = self.passRet[i]["T"]
                Tmax_ = T_[absIndexMax_]

                # position
                x_PTmax = time[-1] * 0.9
                y_PTmax = Tmax_ * 0.8
                if time[-1] * 0.1 <= absIndexMax_ <= time[-1] * 0.9:
                    x_Tmax = absIndexMax_
                elif absIndexMax_ > time[-1] * 0.9:
                    x_Tmax = absIndexMax_ - time[-1] * 0.1
                else:
                    x_Tmax = absIndexMax_ + time[-1] * 0.1
                y_Tmax = Tmax_ * 0.2

                # plot
                draw_T = draw_plot.Draw(fig.ax[i + 1], fig, gridy=True, title=None,
                                        labelx="Time" if (i == n - 1) else None,
                                        labely="T", ylim=[0, Tmax_ * 1.1], xlim=[0, time[-1]])
                line_T = draw_plot.PlotDraw(time, T_, color="k", linewidth=0.6)
                Text_PTmax = draw_plot.TextDraw("PTmax: " + '%.2f' % PTmax_, [x_PTmax, y_PTmax])
                Text_Tmax = draw_plot.TextDraw("Tmax in " + str(absIndexMax_), [x_Tmax, y_Tmax], color="r")
                line_Tmax = draw_plot.PlotDraw([absIndexMax_, absIndexMax_], [0, Tmax_], linestyle="--", color="r",
                                               linewidth=0.6)
                draw_T.adddraw(line_T)
                draw_T.adddraw(Text_PTmax)
                draw_T.adddraw(Text_Tmax)
                draw_T.adddraw(line_Tmax)

        # len(passRet)==0: namely, self._data could not be split, there is no breakpoint
        else:
            # plot data and mean lines
            fig = draw_plot.FigureVert(n + 1)
            draw_data = draw_plot.Draw(fig.ax, fig, gridy=True, title="BG Variation detect", labelx="Time",
                                       labely=labely)
            line_data = draw_plot.PlotDraw(self._data, alpha=0.6, color="gray", linewidth=2)
            line_mean = draw_plot.PlotDraw(sub_mean, color="k", linewidth=0.6)

            draw_data.adddraw(line_data)
            draw_data.adddraw(line_mean)

        # set ticks while time_ticks!=None
        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])

    @staticmethod
    def split(data, confidence: float = 0.95, l=25):
        ''' split function: split based on max T
        input:
            data: 1D array, the time series to split
            confidence: statistic confidence, default=0.95
            l: the min length to split, l should equal or greater than 25, default=25

        output:
            T: T statistic series, len=len(data), T[0]=T[-1]=0 where T does not calculate
            indexMax: the Tmax index in data, it is a relative index
            PTmax: probability for T <= Tmax, do split when PTmax >= confidence(or 1 - PTmax <= alpha)
            pass_: bool, it is True while PTmax >= confidence and len(data) > l

        '''

        # len < l, return pass_ = False
        if len(data) <= l:
            return None, None, None, False

        T = np.zeros_like(data, dtype=float)

        for i in range(1, len(data) - 1):
            ''' loop to cal T[i], which does not contain the first/last point '''
            left = data[:i + 1]
            right = data[i:]
            s1 = np.std(left, ddof=1)
            s2 = np.std(right, ddof=1)
            u1 = np.mean(left)
            u2 = np.mean(right)
            n1 = len(left)
            n2 = len(right)
            sd = ((((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 + n2 - 2)) ** 0.5) * ((1 / n1 + 1 / n2) ** 0.5)
            T[i] = abs((u1 - u2) / sd)

        # index of T where T == Tmax
        indexMax = np.argmax(T)

        # cal PTmax
        eta = 4.19 * math.log(len(data)) - 11.54
        Delta = 0.40
        v = len(data) - 2
        c = v / (v + max(T) ** 2)
        PTmax = (1 - (betainc(Delta * v, Delta, c))) ** eta

        # while PTmax >= confidence and len(data) > l(judged it before), it could be split (==True)
        pass_ = (PTmax >= confidence)

        return T, indexMax, PTmax, pass_

    @staticmethod
    def generateSub(subSplits: list, confidence, l):
        ''' generator to generate split Sub time series
        input:
            subSplits: init subSplits, default = [{"absIndex": absIndex, "subSplit": self._data}], which contains subs need
                       to split
            confidence, l: params used in BGVD.split()

        output:
            {"subSplit": subSplit, "absIndex": absIndex,"absIndexMax": indexMax, "relativeIndexMax": indexMax_, "T": T,
             "PTmax": PTmax, "pass_": pass_}
            subSplit: the sub time series that will be split
            absIndex: absIndex array cut to subSplit series, len=len(subSplit)
            absIndexMax / relativeIndexMax: absolute and relative index for max T
            T, relativeIndexMax, PTmax, pass_: result from executing function BGVD.split(subSplit), belong to subSplit
        '''

        while True:
            # return condition: no subSplits need to split
            if len(subSplits) == 0:
                return

            # loop to generate split() result
            for i in range(len(subSplits)):
                pop_ = subSplits.pop()  # pop the last element to split
                subSplit = pop_["subSplit"]
                absIndex = pop_["absIndex"]
                T, relativeIndexMax, PTmax, pass_ = BGVD.split(subSplit, confidence=confidence, l=l)  # split
                absIndexMax = absIndex[relativeIndexMax]  # trans relative index into absolute index
                yield {"subSplit": subSplit, "absIndex": absIndex, "absIndexMax": absIndexMax, "relativeIndexMax":
                    relativeIndexMax, "T": T, "PTmax": PTmax, "pass_": pass_}  # yield ret

                # if pass_ == True, namely subSplit can be split, split it to two subs which then append into subSplits
                if pass_ == True:
                    subSplits.append(
                        {"absIndex": absIndex[: relativeIndexMax + 1], "subSplit": subSplit[: relativeIndexMax + 1]})
                    subSplits.append({"absIndex": absIndex[relativeIndexMax:], "subSplit": subSplit[relativeIndexMax:]})


class CCVD(VDBase):
    ''' CCVD, the Cumulative Curve method to detect variation '''

    def __init__(self, data, filter=None):
        ''' init function
        input:
            data: 1D array, time series
            filter: filter to filtering cum data
        '''
        self._data = data
        self.cumdata = self.cum(self._data)
        self.filter = filter

        if self.filter != None:
            self.cumdata = self.filtering(self.cumdata)

        self.slope = self.slope(self.cumdata)
        self.ret = self.detect()

    def detect(self):
        ''' Implement VDBase.detect
        how to detect:
            data -> cum() [-> filtering()] -> slope() -> sort slope and its index
        '''
        index_ = np.arange(len(self._data))
        slope = self.slope()
        sortedindex = sorted(index_, key=lambda index: slope[index], reverse=True)
        sortedslope = sorted(slope, reverse=True)
        ret = {"index": sortedindex, "slope": sortedslope}
        return ret

    def plot(self, time_ticks=None, labely="Cumulative Value", **kwargs):
        ''' Implement VDBase.plot
        input:
            time_ticks: dict {"ticks": ticks, "interval": interval}, the ticks of time, namely axis x
                        note: for ticks: len=len(data), for interval: dtype=int
            labely: the labely
            **kwargs: keyword args of subplots, keyword args in Figure init function
        '''
        # define time
        time = np.arange(len(self._data))

        # plot
        fig = draw_plot.Figure(**kwargs)
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="Cumulative Curve Variation Detect", labely=labely,
                              xlim=[0, time[-1]])

        # original and filter line
        if self.filter != None:
            # plot original data
            line_original_cumdata = draw_plot.PlotDraw(time, self.cum(self._data), alpha=0.6, color="gray", linewidth=2,
                                                    label="Cum Data")
            # plot filter
            line_filter_cumdata = draw_plot.PlotDraw(time, self.cumdata, alpha=0.6, color="gray", linewidth=2,
                                                    label="Filter Cum Data")

        else:
            # plot original data
            line_original_cumdata = draw_plot.PlotDraw(time, self.cumdata, color="k", linewidth=1, label="Cum Data")

        #


        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])

    @staticmethod
    def cum(data):
        ''' Calculate Cumulative series based on data '''
        return np.cumsum(data)

    @staticmethod
    def filtering(data, filter):
        ''' filter to smooth data '''
        pass

    @staticmethod
    def slope(data):
        ''' Calculate slope series based on data
            note: it isn't a instantaneous slope, it is also a split process
        '''

        slope = np.zeros_like(data, dtype=float)

        for i in range(1, len(data) - 1):
            ''' loop to cal slope[i], which does not contain the first/last point '''
            left = data[:i + 1]
            right = data[i:]
            polycurve_left = curve_fit.PolyCurve(np.arange(i+1), left, deg=1)
            polycurve_right = curve_fit.PolyCurve(np.arange(i, len(data)+1), right, deg=1)



        return


if __name__ == '__main__':
    x = np.hstack((np.random.rand(100, ) * 50, np.random.rand(100, ) * 100))
    # x = np.random.rand(1000, )
    # x = sorted(np.random.rand(1000, ))
    # x = np.arange(100)
    # x = np.hstack((np.arange(1000), np.arange(2000, 1000, -1)))
    bgvd = BGVD(x)
    ret = bgvd.passRet
    bp = bgvd.bp
    bgvd.plot()  # time_ticks={"ticks": np.arange(10, 1010), "interval": 100}
