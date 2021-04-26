# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# variation detect
# note: each vd method have its own limits and application scopes, choice method based on your data characteristics
# future work: delete some wrong bp, combine some bps which closed to each other too much.

import numpy as np
import abc
import draw_plot
import math
from scipy.special import betainc
from matplotlib import pyplot as plt
import curve_fit
from scipy import stats
import useful_func
import filter


class VDBase(abc.ABC):
    ''' VDBase abstract class '''

    @abc.abstractmethod
    def detect(self):
        ''' define detect abstarct method '''

    @abc.abstractmethod
    def plot(self):
        ''' define plot abstarct method '''

    def plotdata(self, maxplotnumber=1, time_ticks=None, labelx="Time", labely="Data", **kwargs):
        ''' plot original data '''
        # define time
        time = np.arange(len(self._data))

        # plot original data
        fig = draw_plot.Figure(**kwargs)
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="Data time series", labely=labely,
                              labelx=labelx, xlim=[0, time[-1]], legend_on=False)
        scatter_original_data = draw_plot.ScatterDraw(time, y, marker='o', c="lightgray", edgecolor="gray", s=2)
        draw.adddraw(scatter_original_data)

        # plot bp
        n = maxplotnumber

        for i in range(n):

            if isinstance(self, MKVD):
                # extract
                index_bp = self.bp[i]
                r = useful_func.intersection([int(index_bp), self._data[int(index_bp)], int(index_bp) + 1,
                                              self._data[int(index_bp) + 1]],
                                             [index_bp, self._data[int(index_bp)], index_bp,
                                              self._data[int(index_bp) + 1]])
                y_bp = r[1]

                # text position
                if time[-1] * 0.1 <= index_bp <= time[-1] * 0.9:
                    x_bp = int(index_bp)
                elif index_bp > time[-1] * 0.9:
                    x_bp = int(index_bp) - time[-1] * 0.1
                else:
                    x_bp = int(index_bp) + time[-1] * 0.1

                y_bp = y_bp

                # plot bp
                index_bp_text = "bp in " + ("%.1f" % index_bp if time_ticks == None else "%.1f" % index_bp +
                                " between\n" + str(time_ticks["ticks"][int(index_bp)]) + " and " +
                                str(time_ticks["ticks"][int(index_bp) + 1]))

                scatter_bp = draw_plot.ScatterDraw(index_bp, y_bp, marker='o', c="r", edgecolor="r", s=2)
                Text_bp = draw_plot.TextDraw(index_bp_text, [index_bp, y_bp], color="r")

            else:
                # extract
                index_bp = self.bp[i]
                y_bp = self._data[index_bp]

                # text position
                if time[-1] * 0.1 <= index_bp <= time[-1] * 0.9:
                    x_bp = index_bp
                elif index_bp > time[-1] * 0.9:
                    x_bp = index_bp - time[-1] * 0.1
                else:
                    x_bp = index_bp + time[-1] * 0.1

                y_bp = y_bp

                # plot bp
                index_bp_text = "bp in " + (
                    str(index_bp) if time_ticks == None else str(time_ticks["ticks"][int(index_bp)]))
                scatter_bp = draw_plot.ScatterDraw(index_bp, y_bp, marker='o', c="r", edgecolor="r", s=2)
                Text_bp = draw_plot.TextDraw(index_bp_text, [x_bp, y_bp], color="r")

            draw.adddraw(scatter_bp)
            draw.adddraw(Text_bp)

        # set ticks while time_ticks!=None
        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])


class BGVD(VDBase):
    ''' BGVD, the BG method to detect variation, which depend on the variation between mean of sub-series

        reference: [1] Bernaola-Galván P., Ivanov P.C., Amaral L.A.N. et al. Scale invariance in the nonstationarity of
        human heart rate[J]. Physical review letters, 2001, 87(16): 168105.

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

    def plot(self, maxplotnumber=6, time_ticks=None, labelx="Time", labely="Data", **kwargs):
        ''' Implement VDBase.plot
        input:
            maxplotnumber: max number of plot breakpoint(ax)
            time_ticks: dict {"ticks": ticks, "interval": interval}, the ticks of time, namely axis x
                        note: for ticks: len=len(data), for interval: dtype=int
            labelx/y: the labelx/y of the first ax

            **kwargs: keyword args of subplots, keyword args in Figure init function
        '''

        # define time
        time = np.arange(len(self._data))

        # define sub mean lines
        sorted_bp = sorted(self.bp)
        sorted_bp.insert(0, int(0))
        sorted_bp.append(len(self._data))
        sub_mean = np.zeros((len(self._data),))
        sub_time = np.arange(len(self._data))

        for i in range(len(sorted_bp) - 1):
            slice_ = slice(sorted_bp[i], sorted_bp[i + 1], 1)
            sub_mean[slice_] = np.mean(self._data[slice_])

        # plot
        n = len(self.passRet) if len(self.passRet) <= maxplotnumber else maxplotnumber  # define max fig number = 6
        if n > 0:
            # plot data
            fig = draw_plot.FigureVert(n + 1, sharex=True, **kwargs)
            draw_data = draw_plot.Draw(fig.ax[0], fig, gridy=True, title="BG Variation Detect", labely=labely,
                                       xlim=[0, time[-1]])
            line_data = draw_plot.PlotDraw(time, self._data, alpha=0.6, color="gray", linewidth=2)
            line_mean = draw_plot.PlotDraw(sub_time, sub_mean, color="k", linewidth=0.6)
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

                # text position
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
                                        labelx=labelx if (i == n - 1) else None,
                                        labely="T", ylim=[0, Tmax_ * 1.1], xlim=[0, time[-1]])
                line_T = draw_plot.PlotDraw(time, T_, color="k", linewidth=0.6)
                Text_PTmax = draw_plot.TextDraw("PTmax: " + '%.2f' % PTmax_, [x_PTmax, y_PTmax])
                index_bp_text = (str(absIndexMax_) if time_ticks == None else str(time_ticks["ticks"][absIndexMax_]))
                Text_Tmax = draw_plot.TextDraw("Tmax in " + index_bp_text, [x_Tmax, y_Tmax], color="r")
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
            draw_data = draw_plot.Draw(fig.ax, fig, gridy=True, title="BG Variation detect", labelx=labelx,
                                       labely=labely, xlim=[0, time[-1]])
            line_data = draw_plot.PlotDraw(time, self._data, alpha=0.6, color="gray", linewidth=2)
            line_mean = draw_plot.PlotDraw(sub_time, sub_mean, color="k", linewidth=0.6)

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
            left = data[:i + 1]  # left contains point at i
            right = data[i:]  # right also contains point at i
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
                    # left contains point at i, right also contains point at i
                    subSplits.append({"absIndex": absIndex[relativeIndexMax:], "subSplit": subSplit[relativeIndexMax:]})


class SCCVD(VDBase):
    ''' SCCVD, the Single Cumulative Curve method to detect variation, which depend on the variation between increase
     rate of sub-series '''

    def __init__(self, data, filter=None, constraint=1, l=5, **kwargs):
        ''' init function
        input:
            data: 1D array, time series
            constraint: constraint to limit selecting breakpoints, relative residual diff < constraint (range
                        normalization), default=0.5, 1 means no limit residual diff
            l: length limit that do not calculate slope(set 0), the left [:l] and right [-l:] set 0, default=25, it can
               help us exclude some impossible bp in left and right
            filter: FilterBase' subclass, filter to filtering cum data, "filter" is a class rather than a objection

            **kwargs: keyword args used in filtering()

        output:
            self.ret: dict, {"index", "slope_diff_"}, contains the index and slope_diff_ of each breakpoint, sorted by
                      slope_diff_
            self.bp: the index of the breakpoint which extract from self.ret, sorted by slope_diff_
            self.slope_bp_diff: the slope_diff of the bp which extract from self.ret, sorted by slope_diff_

        '''
        self._data = data
        self.cumdata = self.cum(self._data)
        self.filter = filter
        self.constraint = constraint
        self.l = l

        # flitering
        if self.filter != None:
            self.cumdata = self.filtering(self.cumdata, self.filter, **kwargs)

        self.slope_ret = self.slope(self.cumdata, self.constraint, self.l)  # slope_ret before sort

        # there is some results containing information you want
        self.ret = self.detect()
        self.bp = self.ret["index"]  # bp after sort
        self.slope_bp_diff = [ret["slope_diff_"] for ret in self.ret["slope_ret"]]  # slope_bp_diff after sort

    def detect(self):
        ''' Implement VDBase.detect
        how to detect:
            data -> cum() [-> filtering()] -> slope() -> sort slope and its index
        '''
        # sort slope and index based on abs
        index_ = np.arange(len(self._data))
        slope_ret_ = self.slope_ret
        sortedindex = sorted(index_, key=lambda index: abs(slope_ret_[index]["slope_diff_"]), reverse=True)
        sortedslope_ret = sorted(slope_ret_, key=lambda slope_ret: abs(slope_ret["slope_diff_"]), reverse=True)

        # remove elements in sortedindex & sortedslope_ret which has "slope_diff_" == 0
        index_zeros = [i for i in range(len(sortedslope_ret)) if sortedslope_ret[i]["slope_diff_"] == 0]
        sortedindex = [sortedindex[i] for i in range(len(sortedindex)) if i not in index_zeros]
        sortedslope_ret = [sortedslope_ret[i] for i in range(len(sortedslope_ret)) if i not in index_zeros]

        ret = {"index": sortedindex, "slope_ret": sortedslope_ret}
        return ret

    def plot(self, maxplotnumber=1, time_ticks=None, labelx="Time", labely="Cumulative Value", **kwargs):
        ''' Implement VDBase.plot
        input:
            maxplotnumber: max number of plot breakpoint
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
                              labelx=labelx, xlim=[0, time[-1]], ylim=[0, self.cumdata[-1] * 1.1], legend_on=
                              {"loc": "upper left", "framealpha": 0.8})

        # original and filter line
        if self.filter != None:
            # plot original data
            line_original_cumdata = draw_plot.ScatterDraw(time, self.cum(self._data), marker='o', c="gray",
                                                          edgecolor="gray", s=3, label="Cum Data")
            # plot filter
            line_filter_cumdata = draw_plot.PlotDraw(time, self.cumdata, color="k", linewidth=2,
                                                     label="Filter Cum Data", alpha=0.6)
            draw.adddraw(line_original_cumdata)
            draw.adddraw(line_filter_cumdata)

        else:
            # plot original data
            line_original_cumdata = draw_plot.ScatterDraw(time, self.cumdata, marker='o', c="lightgray",
                                                          edgecolor="lightgray", s=3, label="Cum Data")
            draw.adddraw(line_original_cumdata)

        # plot breakpoint
        n = len(self.bp) if len(self.bp) <= maxplotnumber else maxplotnumber
        for i in range(n):
            # extract variable
            index_bp = self.ret["index"][i]
            slope_diff = self.ret["slope_ret"][i]["slope_diff_"]
            cum_bp = self.cumdata[index_bp]
            pcf_left = self.ret["slope_ret"][i]["pcf_left"]
            pcf_right = self.ret["slope_ret"][i]["pcf_right"]

            # text position
            if time[-1] * 0.1 <= index_bp <= time[-1] * 0.9:
                x_bp = index_bp
            elif index_bp > time[-1] * 0.9:
                x_bp = index_bp - time[-1] * 0.01
            else:
                x_bp = index_bp + time[-1] * 0.01
            # x_bp = index_bp
            y_bp = cum_bp * (1 - i / n)

            # alpha
            alpha_ = 1 - i / n
            fontdict = {"family": "Arial", "size": 5}

            # plot left/right line
            time_left = time[:index_bp + 1]  # left contains point at i
            time_right = time[index_bp:]  # right also contains point at i
            cumfit_left = np.polyval(np.poly1d(pcf_left), time_left)
            cumfit_right = np.polyval(np.poly1d(pcf_right), time_right)

            line_left = draw_plot.PlotDraw(time_left, cumfit_left, alpha=alpha_, color="b", linewidth=0.6,
                                           label="left fit line" if i == 0 else None, linestyle="-")
            line_right = draw_plot.PlotDraw(time_right, cumfit_right, alpha=alpha_, color="r",
                                            linewidth=0.6,
                                            label="right fit line" if i == 0 else None, linestyle="-")

            draw.adddraw(line_left)
            draw.adddraw(line_right)

            # plot bp
            index_bp_text = str(index_bp) if time_ticks == None else str(time_ticks["ticks"][index_bp])
            line_bp = draw_plot.PlotDraw([index_bp, index_bp], [0, cum_bp], alpha=alpha_, linestyle="--", color="r",
                                         linewidth=0.6,
                                         label=f"bp{i}:(slope diff={'%.2f' % slope_diff}, index={index_bp_text})")
            Text_bp = draw_plot.TextDraw(f"bp{i}", [x_bp, y_bp], color="k", fontdict=fontdict, zorder=20)

            draw.adddraw(line_bp)
            draw.adddraw(Text_bp)

        # set ticks while time_ticks!=None
        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])

    @staticmethod
    def cum(data):
        ''' Calculate Cumulative series based on data '''
        return np.cumsum(data)

    @staticmethod
    def filtering(data, filter, **kwargs):
        ''' filter to smooth data '''
        flt = filter(data, **kwargs)
        return flt.filtered_data

    @staticmethod
    def slope(data, constraint, l):
        ''' Calculate slope series based on data
            input:
                data: 1D array, the time series to split and calculate slope_diff
                constraint: constraint to limit selecting breakpoints, relative diff < constraint (range normalization)
                l: length limit that do not calculate slope(set 0), the left [:l] and right [-l:] set 0

            note: it isn't a instantaneous slope, it is also a split process

            output:
                slope_ret: list, which contain [{"slope_diff_": slope_diff_, "pcf_left": pcf_left, "pcf_right":
                                                pcf_right}...] for each point except the first/last point
                           slope_diff: slope_left - slope_right, >0: downtrend, <0: uptrend, abs(slope_diff) determine
                                       rank of breakpoint
        '''

        # define slope_ret
        slope_ret = []
        residuals_diff = []

        # left l which not calculate
        slope_ret.extend([{"slope_diff_": 0, "pcf_left": 0, "pcf_right": 0}] * l)
        residuals_diff.extend([0] * l)

        # [:] - [:l] - [-l:] = [l: -l] to calculate slope
        for i in range(l, len(data) - l):
            ''' loop to cal slope[i], which does not contain the first/last point '''
            left = data[:i + 1]  # left contains point at i
            right = data[i:]  # right also contains point at i
            pc_left = curve_fit.PolyCurve(np.arange(i + 1), left, show=False, deg=1)
            pc_right = curve_fit.PolyCurve(np.arange(i, len(data)), right, show=False, deg=1)
            pcf_left = pc_left.pcf
            pcf_right = pc_right.pcf

            # add constraint to limit selecting breakpoints: residuals != []
            slope_diff_ = 0  # init slope_diff_== 0
            # if it passes constraint, slope_diff_ = pcf_left[0] - pcf_right[0]
            if len(pc_left.residuals) != 0 and len(pc_right.residuals) != 0:
                left_residuals = pc_left.residuals[0]
                right_residuals = pc_right.residuals[0]
                residuals_diff.append(abs(left_residuals - right_residuals))
                slope_diff_ = pcf_left[0] - pcf_right[0]

            slope_ret.append({"slope_diff_": slope_diff_, "pcf_left": pcf_left, "pcf_right": pcf_right})

        # right l which not calculate
        slope_ret.extend([{"slope_diff_": 0, "pcf_left": 0, "pcf_right": 0}] * l)
        residuals_diff.extend([0] * l)

        # add constraint to limit selecting breakpoint: residual relative diff < constraint based on range normalization
        # relative diff = (diff - min(diff)) / (max(diff) - min(diff)) < constraint, where diff = abs(left_residuals -
        # right_residuals)
        residuals_relative_diff = [(diff - min(residuals_diff)) / (max(residuals_diff) - min(residuals_diff)) for diff
                                   in residuals_diff]
        remove_index = [i for i in range(len(residuals_relative_diff)) if residuals_relative_diff[i] > constraint]
        if len(remove_index) > 0:
            for i in range(len(remove_index)):
                slope_ret[i]["slope_diff_"] = 0
                slope_ret[i]["pcf_left"] = 0
                slope_ret[i]["pcf_right"] = 0

        return slope_ret


class DCCVD(SCCVD):
    ''' DCCVD, the Double Cumulative Curve method to detect variation, which depend on the variation between increase
         rate of sub-series '''

    def __init__(self, data1, data2, filter=None, constraint=1, l=5, **kwargs):
        ''' init function
        input:
            similar with SCCVD, but introduce data2
            data1: x axis of the cum curve
            data2: y axis of the cum curve

        output:
            similar with SCCVD
        '''
        self._data1 = data1
        self._data2 = data2
        self.cumdata1 = self.cum(self._data1)
        self.cumdata2 = self.cum(self._data2)
        self.filter = filter
        self.constraint = constraint
        self.l = l

        # flitering
        if self.filter != None:
            self.cumdata1, self.cumdata2 = self.filtering(self.cumdata1, self.cumdata2, self.filter, **kwargs)

        self.slope_ret = self.slope(self.cumdata1, self.cumdata2, self.constraint, self.l)  # slope_ret before sort

        # there is some results containing information you want
        self.ret = self.detect()
        self.bp = self.ret["index"]  # bp after sort
        self.slope_bp_diff = [ret["slope_diff_"] for ret in self.ret["slope_ret"]]  # slope_bp_diff after sort

    def detect(self):
        ''' Implement VDBase.detect '''
        # sort slope and index based on abs
        index_ = np.arange(len(self._data1))
        slope_ret_ = self.slope_ret
        sortedindex = sorted(index_, key=lambda index: abs(slope_ret_[index]["slope_diff_"]), reverse=True)
        sortedslope_ret = sorted(slope_ret_, key=lambda slope_ret: abs(slope_ret["slope_diff_"]), reverse=True)

        # remove elements in sortedindex & sortedslope_ret which has "slope_diff_" == 0
        index_zeros = [i for i in range(len(sortedslope_ret)) if sortedslope_ret[i]["slope_diff_"] == 0]
        sortedindex = [sortedindex[i] for i in range(len(sortedindex)) if i not in index_zeros]
        sortedslope_ret = [sortedslope_ret[i] for i in range(len(sortedslope_ret)) if i not in index_zeros]

        ret = {"index": sortedindex, "slope_ret": sortedslope_ret}
        return ret

    def plot(self, maxplotnumber=1, labelx="Cumulative Value", labely="Cumulative Value", **kwargs):
        ''' Implement VDBase.plot '''
        # define time
        x = self.cumdata1
        y = self.cumdata2

        # plot
        fig = draw_plot.Figure(**kwargs)
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="Cumulative Curve Variation Detect", labely=labely,
                              labelx=labelx, xlim=[0, x[-1]], ylim=[0, y[-1] * 1.1], legend_on=
                              {"loc": "upper left", "framealpha": 0.8})

        # original and filter line
        if self.filter != None:
            # plot original data
            line_original_cumdata = draw_plot.ScatterDraw(self.cum(self._data1), self.cum(self._data2), marker='o',
                                                          c="gray", edgecolor="gray", s=3, label="Cum Data")
            # plot filter
            line_filter_cumdata = draw_plot.PlotDraw(x, y, color="k", linewidth=2, label="Filter Cum Data", alpha=0.6)
            draw.adddraw(line_original_cumdata)
            draw.adddraw(line_filter_cumdata)

        else:
            # plot original data
            line_original_cumdata = draw_plot.ScatterDraw(x, y, marker='o', c="lightgray",
                                                          edgecolor="lightgray", s=3, label="Cum Data")
            draw.adddraw(line_original_cumdata)

        # plot breakpoint
        n = len(self.bp) if len(self.bp) <= maxplotnumber else maxplotnumber
        for i in range(n):
            # extract variable
            index_bp = self.ret["index"][i]
            slope_diff = self.ret["slope_ret"][i]["slope_diff_"]
            cumx_bp = x[index_bp]
            cumy_bp = y[index_bp]
            pcf_left = self.ret["slope_ret"][i]["pcf_left"]
            pcf_right = self.ret["slope_ret"][i]["pcf_right"]

            # text position
            if x[-1] * 0.1 <= index_bp <= x[-1] * 0.9:
                x_bp = cumx_bp
            elif index_bp > x[-1] * 0.9:
                x_bp = cumx_bp - x[-1] * 0.01
            else:
                x_bp = cumx_bp + x[-1] * 0.01
            # x_bp = index_bp
            y_bp = cumy_bp * (1 - i / n)

            # alpha
            alpha_ = 1 - i / n
            fontdict = {"family": "Arial", "size": 5}

            # plot left/right line
            leftx = x[:index_bp + 1]  # left contains point at i
            rightx = x[index_bp:]  # right also contains point at i
            cumfit_left = np.polyval(np.poly1d(pcf_left), leftx)
            cumfit_right = np.polyval(np.poly1d(pcf_right), rightx)

            line_left = draw_plot.PlotDraw(leftx, cumfit_left, alpha=alpha_, color="b", linewidth=0.6,
                                           label="left fit line" if i == 0 else None, linestyle="-")
            line_right = draw_plot.PlotDraw(rightx, cumfit_right, alpha=alpha_, color="r",
                                            linewidth=0.6,
                                            label="right fit line" if i == 0 else None, linestyle="-")

            draw.adddraw(line_left)
            draw.adddraw(line_right)

            # plot bp
            line_bp = draw_plot.PlotDraw([cumx_bp, cumx_bp], [0, cumy_bp], alpha=alpha_, linestyle="--", color="r",
                                         linewidth=0.6,
                                         label=f"bp{i}:(slope diff={'%.2f' % slope_diff}, index={index_bp})")
            Text_bp = draw_plot.TextDraw(f"bp{i}", [x_bp, y_bp], color="k", fontdict=fontdict, zorder=20)

            draw.adddraw(line_bp)
            draw.adddraw(Text_bp)

    @staticmethod
    def filtering(data1, data2, filter, **kwargs):
        ''' filter to smooth data '''
        data = np.vstack((data1, data2))
        flt = filter(data, **kwargs)
        return flt.filtered_data[0, :], flt.filtered_data[1, :]

    @staticmethod
    def slope(data1, data2, constraint, l):
        ''' Calculate slope series based on data: similar with SCCVD, but data1 is x axis, data2 is y axis '''
        # define slope_ret
        slope_ret = []
        residuals_diff = []

        # left l which not calculate
        slope_ret.extend([{"slope_diff_": 0, "pcf_left": 0, "pcf_right": 0}] * l)
        residuals_diff.extend([0] * l)

        # [:] - [:l] - [-l:] = [l: -l] to calculate slope
        for i in range(l, len(data1) - l):
            ''' loop to cal slope[i], which does not contain the first/last point '''
            leftx = data1[:i + 1]  # left contains point at i
            rightx = data1[i:]  # left contains point at i
            lefty = data2[:i + 1]
            righty = data2[i:]  # right also contains point at i
            pc_left = curve_fit.PolyCurve(leftx, lefty, show=False, deg=1)
            pc_right = curve_fit.PolyCurve(rightx, righty, show=False, deg=1)
            pcf_left = pc_left.pcf
            pcf_right = pc_right.pcf

            # add constraint to limit selecting breakpoints: residuals != []
            slope_diff_ = 0  # init slope_diff_== 0
            # if it passes constraint, slope_diff_ = pcf_left[0] - pcf_right[0]
            if len(pc_left.residuals) != 0 and len(pc_right.residuals) != 0:
                left_residuals = pc_left.residuals[0]
                right_residuals = pc_right.residuals[0]
                residuals_diff.append(abs(left_residuals - right_residuals))
                slope_diff_ = pcf_left[0] - pcf_right[0]

            slope_ret.append({"slope_diff_": slope_diff_, "pcf_left": pcf_left, "pcf_right": pcf_right})

        # right l which not calculate
        slope_ret.extend([{"slope_diff_": 0, "pcf_left": 0, "pcf_right": 0}] * l)
        residuals_diff.extend([0] * l)

        # add constraint to limit selecting breakpoint: residual relative diff < constraint based on range normalization
        # relative diff = (diff - min(diff)) / (max(diff) - min(diff)) < constraint, where diff = abs(left_residuals -
        # right_residuals)
        residuals_relative_diff = [(diff - min(residuals_diff)) / (max(residuals_diff) - min(residuals_diff)) for diff
                                   in residuals_diff]
        remove_index = [i for i in range(len(residuals_relative_diff)) if residuals_relative_diff[i] > constraint]
        if len(remove_index) > 0:
            for i in range(len(remove_index)):
                slope_ret[i]["slope_diff_"] = 0
                slope_ret[i]["pcf_left"] = 0
                slope_ret[i]["pcf_right"] = 0

        return slope_ret


class MKVD(VDBase):
    ''' MKVD, the MK method to detect variation, which depend on the variation between mean of sub-series
    '''

    def __init__(self, data, confidence: float = 0.95):
        ''' init function
        input:
            data: 1D array, time series
            confidence: statistic confidence, default=0.95

        output:
            self.interP: list, [{"index", "U"}...], All points where UF cross UB
            self.bp: list, [{"index", "U"}...], points where UB cross UB and passed the significance test

        '''
        self._data = data
        self.confidence = confidence
        self.alpha = 1 - self.confidence
        self.u0 = abs(stats.norm.ppf(self.alpha / 2))
        self.UF, self.UB, self.interP, self.bp_all = self.detect()
        self.bp = [bp["index"] for bp in self.bp_all]

    def detect(self):
        ''' Implement VDBase.detect
        how to detect:
            data -> calStatisticsU -> UF/UK -> intersectionPoint -> interp -> significant test -> bp
        '''
        UF = self.calStatisticsU(self._data)
        UB = [ub * -1 for ub in self.calStatisticsU(self._data[::-1])[::-1]]
        interP = self.intersectionPoint(UF, UB)
        bp = [ip for ip in interP if abs(ip["U"]) < self.u0]
        return UF, UB, interP, bp

    def plot(self, time_ticks=None, labelx="Time", labely="UF/UB", **kwargs):
        ''' Implement VDBase.plot
        input:
            time_ticks: dict {"ticks": ticks, "interval": interval}, the ticks of time, namely axis x
                        note: for ticks: len=len(data), for interval: dtype=int
            labelx/y: the labelx/y of the first ax

            **kwargs: keyword args of subplots, keyword args in Figure init function
        '''
        # define time
        time = np.arange(len(self._data))

        # plot
        expand = (max(max(self.UF), max(self.UB)) - min(min(self.UF), min(self.UB))) * 0.05
        ylim = [min(min(self.UF), min(self.UB)) - expand, max(max(self.UF), max(self.UB)) + expand]
        fig = draw_plot.Figure(**kwargs)
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="MK Variation Detect", labely=labely,
                              labelx=labelx, xlim=[0, time[-1]], ylim=ylim,
                              legend_on={"loc": "upper left", "framealpha": 0.8})

        # UF UB line
        line_UF = draw_plot.PlotDraw(time, self.UF, color="k", linewidth=0.6, label="UF")
        line_UB = draw_plot.PlotDraw(time, self.UB, color="gray", linewidth=0.6, label="UB")

        # adddraw
        draw.adddraw(line_UF)
        draw.adddraw(line_UB)

        # u0
        line_u0plus = draw_plot.PlotDraw(time, np.full((len(time),), fill_value=self.u0), "b-",
                                         label=f"u0: " + "%.2f" % self.u0,
                                         linewidth=0.6)
        line_u0minus = draw_plot.PlotDraw(time, np.full((len(time),), fill_value=-self.u0), "b-", linewidth=0.6)

        # adddraw
        draw.adddraw(line_u0plus)
        draw.adddraw(line_u0minus)

        # bp
        if len(self.bp_all) > 0:
            for i in range(len(self.bp_all)):
                # extract
                index_bp = self.bp_all[i]["index"]
                U_bp = self.bp_all[i]["U"]

                # text position
                if time[-1] * 0.1 <= index_bp <= time[-1] * 0.9:
                    x_bp = index_bp
                elif index_bp > time[-1] * 0.9:
                    x_bp = index_bp - time[-1] * 0.1
                else:
                    x_bp = index_bp + time[-1] * 0.1

                y_bp = U_bp

                # plot bp
                index_bp_text = "%.1f" % index_bp if time_ticks == None else "%.1f" % index_bp + " between\n" + \
                                str(time_ticks["ticks"][int(index_bp)]) + " and " +\
                                str(time_ticks["ticks"][int(index_bp) + 1])
                line_bp = draw_plot.PlotDraw([index_bp, index_bp], [ylim[0], U_bp], linestyle="--", color="r",
                                             linewidth=0.6)
                Text_bp = draw_plot.TextDraw(index_bp_text, [x_bp, y_bp], color="r")

                # adddraw
                draw.adddraw(line_bp)
                draw.adddraw(Text_bp)

        # set ticks while time_ticks!=None
        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])

    @staticmethod
    def calStatisticsU(vals):
        ''' calculate statistics U
        input:
            vals: data series to do MKVD, there is used to calculate U series

        output:
            U: list, U statistics
        '''

        valslen = len(vals)
        s = []

        # cal s
        for k in range(1, valslen + 1):
            sumval = 0

            for i in range(k):
                for j in range(i):
                    if vals[i] > vals[j]:
                        sumval += 1

            s.append(sumval)

        # cal E(s) and Sde(s) (Standard deviation)
        E = [k * (k + 1) / 4 for k in range(1, valslen + 1)]
        Sde = [(k * (k - 1) * (2 * k + 5) / 72) ** 0.5 for k in range(1, valslen + 1)]

        Sde[0] = 1  # init point 0 to avoid Error in division

        # cal U
        U = [(s[k] - E[k]) / Sde[k] for k in range(valslen)]  # del abs()
        U[0] = 0  # init point 0

        return U

    @staticmethod
    def intersectionPoint(UF, UB):
        ''' Find intersection Point between UF and UB series
        input:
            UF/UB: list, UF & UB series

        output:
            interP: list, [{"index", "U"}...] contains index and U of intersection Points

        '''
        diff = [UF[i] - UB[i] for i in range(len(UF))]
        interP = []
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] == 0:
                interP.append({"index": i, "U": UF[i]})
            elif diff[i] * diff[i + 1] < 0:
                r = useful_func.intersection([i, UF[i], i + 1, UF[i + 1]], [i, UB[i], i + 1, UB[i + 1]])
                interP.append({"index": r[0], "U": r[1]})

        return interP


class OCVD(VDBase):
    ''' OCVD, the Ordered Clustering method to detect variation the min value of sum of squares of deviations between
        sub-series, which breaked by bp
    '''

    def __init__(self, data):
        ''' init function
        input:
            data: 1D array, time series

        output:

        '''

        self._data = data
        self.S, self.sortedindex, self.sortedS = self.detect()
        self.bp = self.sortedindex

    def detect(self):
        ''' Implement VDBase.detect
        how to detect:
            data -> calStatisticS() -> S -> sort, sortWithIndex -> sortedindex, sortedS
        '''

        S = self.calStatisticS(self._data)
        sortedindex, sortedS = useful_func.sortWithIndex(S)

        return S, sortedindex, sortedS

    def plot(self, maxplotnumber=6, time_ticks=None, labelx="Time", labely="S", **kwargs):
        ''' Implement VDBase.plot
        input:
            maxplotnumber: max number of plot breakpoint(ax)
            time_ticks: dict {"ticks": ticks, "interval": interval}, the ticks of time, namely axis x
                        note: for ticks: len=len(data), for interval: dtype=int
            labely: the labely of the first ax

            **kwargs: keyword args of subplots, keyword args in Figure init function
        '''

        # define time
        time = np.arange(len(self._data))

        # plot
        ylim = [min(self.S) * 0.9, max(self.S) * 1.1]
        fig = draw_plot.Figure(**kwargs)
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="Ordered Clustering Variation Detect", labely=labely,
                              labelx=labelx, xlim=[0, time[-1]], ylim=ylim, legend_on=False)

        # S line
        line_S = draw_plot.PlotDraw(time, self.S, color="k", linewidth=0.6)
        draw.adddraw(line_S)

        # bp plot
        n = maxplotnumber

        for i in range(n):
            # extract
            index_bp = self.sortedindex[i]
            S_bp = self.sortedS[i]

            # text position
            if time[-1] * 0.1 <= index_bp <= time[-1] * 0.9:
                x_bp = index_bp
            elif index_bp > time[-1] * 0.9:
                x_bp = index_bp - time[-1] * 0.1
            else:
                x_bp = index_bp + time[-1] * 0.1

            y_bp = S_bp

            # plot bp
            index_bp_text = index_bp if time_ticks == None else time_ticks["ticks"][index_bp]
            line_bp = draw_plot.PlotDraw([index_bp, index_bp], [ylim[0], S_bp], linestyle="--", color="r",
                                         linewidth=0.6)
            Text_bp = draw_plot.TextDraw(f"S = " + "%.d" % S_bp + f" in {index_bp_text}", [x_bp, y_bp], color="r")

            # adddraw
            draw.adddraw(line_bp)
            draw.adddraw(Text_bp)

        # set ticks while time_ticks!=None
        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])

    @staticmethod
    def calStatisticS(data):
        ''' calculate statistics S
        input:
            data: vals: data series to do OCVD, there is used to calculate S series

        output:
            S: np.array, S statistics
        '''

        n = len(data)
        S = np.zeros((n,))

        # set the S in first & last point
        AllV = sum([(x - sum(data) / n) ** 2 for x in data])
        S[0] = AllV
        S[-1] = AllV

        # loop to calculate S in tau between [1, n - 1]
        for i in range(1, n - 1):
            left = data[: i + 1]
            right = data[i:]
            leftV = sum([(x - sum(left) / len(left)) ** 2 for x in left])
            rightV = sum([(x - sum(right) / len(right)) ** 2 for x in right])
            S[i] = leftV + rightV

        return S


if __name__ == '__main__':
    # set sample x
    x = np.hstack((np.random.rand(100, ) * 10, np.random.rand(100, ) * 100))
    y = np.hstack((np.random.rand(100, ) * 10, np.random.rand(100, ) * 600))
    # x = np.random.rand(1000, )
    # x = sorted(np.random.rand(1000, ))
    # x = np.arange(100)
    # x = np.hstack((np.arange(100), np.arange(200, 100, -1)))
    # x = np.hstack((np.random.rand(100, ) * 10, np.arange(200, 100, -1)))

    # bgvd
    bgvd = BGVD(x)
    ret_bgvd = bgvd.passRet
    bp_bgvd = bgvd.bp
    # bgvd.plot(time_ticks={"ticks": [str(i) + 'x' for i in np.arange(200)], "interval": 10})

    # sccvd
    sccvd = SCCVD(x)
    ret_sccvd = sccvd.ret
    bp_sccvd = sccvd.bp
    # bp_diff_sccvd = sccvd.slope_bp_diff
    # sccvd.plot(5, time_ticks={"ticks": [str(i) + 'x' for i in np.arange(200)], "interval": 10})

    # sccvd with filter
    # flt = filter.ButterFilter
    # sccvdf = SCCVD(x, filter=flt, N=2, Wn=0.3)
    # ret_sccvdf = sccvdf.ret
    # bp_sccvdf = sccvdf.bp
    # bp_diff_sccvdf = sccvdf.slope_bp_diff
    # sccvdf.plot(5, time_ticks={"ticks": [str(i) + 'x' for i in np.arange(200)], "interval": 10})

    # dccvd
    # dccvd = DCCVD(x, y)
    # ret_dccvd = dccvd.ret
    # bp_dccvd = dccvd.bp
    # bp_diff_dccvd = dccvd.slope_bp_diff
    # dccvd.plot(5)  # time_ticks={"ticks": np.arange(190), "interval": 10}

    # dccvd with filter
    # flt = filter.ButterFilter
    # dccvdf = DCCVD(x, y, filter=flt, N=2, Wn=0.3)
    # ret_dccvdf = dccvdf.ret
    # bp_dccvdf = dccvdf.bp
    # bp_diff_dccvdf = dccvdf.slope_bp_diff
    # dccvdf.plot(5)

    # mkvd
    mkvd = MKVD(x)
    interP_mkvd = mkvd.interP
    bp_mkvd = mkvd.bp
    # mkvd.plot(time_ticks={"ticks": [str(i) + 'x' for i in np.arange(200)], "interval": 10})
    # mkvd.plotdata()

    # ocvd
    ocvd = OCVD(x)
    bp_ocvd = ocvd.bp
    # ocvd.plot(maxplotnumber=1, time_ticks={"ticks": [str(i) + 'x' for i in np.arange(200)], "interval": 10})
