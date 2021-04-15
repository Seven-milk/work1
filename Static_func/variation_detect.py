# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# variation detect

import numpy as np
import abc
import draw_plot
import math
from scipy.special import betainc


class VDBase(abc.ABC):
    ''' VDBase abstract class '''

    @abc.abstractmethod
    def detect(self):
        ''' define detect abstarct method '''

    @abc.abstractmethod
    def plot(self):
        ''' define plot abstarct method '''


class BGVD(VDBase):
    ''' BGVD, the BG method to detect variation

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
            self.allRet: detect results, a list contains each split process, it has field "subSplit" "absIndex" "T"
                    "indexMax" "PTmax" "pass_"
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
        self.bp = [ret["indexMax"] for ret in self.allRet if ret["pass_"] == True]

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

    def plot(self, time_ticks=None, labely="data"):
        ''' Implement VDBase.plot '''
        # define sub mean series
        sub_mean = np.zeros((len(self._data),))
        sorted_bp = sorted(self.bp)
        sorted_bp.insert(0, int(0))
        sorted_bp.append(int(-1))
        for i in range(len(sorted_bp) - 1):
            slice_ = slice(sorted_bp[i], sorted_bp[i + 1])
            sub_mean[slice_] = np.mean(self._data[slice_])

        # plot data and mean lines
        fig = draw_plot.FigureVert(len(self.passRet) + 1)
        draw_data = draw_plot.Draw(fig.ax[0], fig, gridy=True, title="BG Variation detect", labelx="t", labely=labely)
        line_data = draw_plot.PlotDraw(self._data, alpha=0.6, color="gray")
        line_mean = draw_plot.PlotDraw(sub_mean, color="k")

        draw_data.adddraw(line_data)
        draw_data.adddraw(line_mean)

        # set ticks while time_ticks!=None

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
            {"subSplit": subSplit, "T": T, "indexMax": indexMax, "PTmax": PTmax, "pass_": pass_}
            subSplit: the sub time series
            T, indexMax, PTmax, pass_: result from executing function BGVD.split(subSplit)
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
                T, indexMax_, PTmax, pass_ = BGVD.split(subSplit, confidence=confidence, l=l)  # split
                indexMax = absIndex[indexMax_]  # trans relative index(indexMax_) into absolute index(indexMax)
                yield {"subSplit": subSplit, "absIndex": absIndex, "T": T, "indexMax": indexMax, "PTmax": PTmax,
                       "pass_": pass_}  # yield ret

                # if pass_ == True, namely subSplit can be split, split it to two subs which then append into subSplits
                if pass_ == True:
                    subSplits.append({"absIndex": absIndex[: indexMax_ + 1], "subSplit": subSplit[: indexMax_ + 1]})
                    subSplits.append({"absIndex": absIndex[indexMax_:], "subSplit": subSplit[indexMax_:]})


if __name__ == '__main__':
    # x = sorted(np.random.rand(100, ))
    x = np.random.rand(100, )
    # x = np.arange(100)
    # x = np.hstack((np.arange(100), np.arange(200, 100, -1)))
    bgvd = BGVD(x)
    ret = bgvd.passRet
    bp = bgvd.bp
    bgvd.plot()
