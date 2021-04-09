# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# mannkendall_test
# https://github.com/manaruchi/MannKendall_Sen_Rainfall
import numpy as np
from scipy import stats
import draw_plot
import curve_fit

class MkTest:
    ''' MannKendall Test '''

    def __init__(self, vals: np.ndarray, confidence: float = 0.95, **kwargs):
        ''' Init function
        input:
            vals: variable, 1D np.ndarray
            confidence: static confidence level
            **kwargs: keyword args, it could contain "x" [for plot]
        output:
            self.mkret = {"z": z, "p": p, "trend": tr_type, "se": se}, z: z statistic value, p: p value, trend: trend
                        under confidence test[0: No significant; 1: Significant upward trend; -1: Significant downward
                        trend], "se" : Var(s) after repetition correction
            self.senret = senret = {"slope": slope, "k": k, "se": se}, slope: slope value, k: k value, se: Var(s) after
                        repetition correction
        '''
        self.vals = vals
        self.confidence = confidence
        self.kwargs = kwargs
        self.len = len(self.vals)
        self.mkret = self.mkTest(self.vals, self.confidence)
        self.senret = self.senSlope(self.vals, self.confidence)

    @staticmethod
    def mkTest(vals, confidence):
        ''' mkTest '''
        valslen = len(vals)
        box = np.ones((valslen, valslen))
        box = box * 5
        sumval = 0

        # cal sumval
        for r in range(valslen):
            for c in range(valslen):
                if (r > c):
                    if (vals[r] > vals[c]):
                        box[r, c] = 1
                        sumval = sumval + 1
                    elif (vals[r] < vals[c]):
                        box[r, c] = -1
                        sumval = sumval - 1
                    else:
                        box[r, c] = 0

        # caluclate frequency for each unique value
        freq = 0
        tp = np.unique(vals, return_counts=True)
        for tpx in range(len(tp[0])):
            if (tp[1][tpx] > 1):
                tp1 = tp[1][tpx]
                sev = tp1 * (tp1 - 1) * (2 * tp1 + 5)
                freq = freq + sev

        # Repetition correction based on freq, se = Var(s) after correction
        se = ((valslen * (valslen - 1) * (2 * valslen + 5) - freq) / 18.0) ** 0.5

        # calculate z value
        if (sumval > 0):
            z = (sumval - 1) / se
        else:
            z = (sumval + 1) / se

        # calculate p value
        p = 2 * stats.norm.cdf(-abs(z))

        # trend type, confidence used to determine whether the confidence test has been passed
        if (p < (1 - confidence) and z < 0):
            tr_type = -1
        elif (p < (1 - confidence) and z > 0):
            tr_type = +1
        else:
            tr_type = 0

        # result of mktest
        mkret = {"z": z, "p": p, "trend": tr_type, "se": se}

        return mkret

    @staticmethod
    def senSlope(vals, confidence):
        ''' senSlope '''
        valslen = len(vals)
        alpha = 1 - confidence
        box = np.ones((valslen, valslen))
        box = box * 5
        boxlist = []

        for r in range(valslen):
            for c in range(valslen):
                if (r > c):
                    box[r, c] = (vals[r] - vals[c]) / (r - c)
                    boxlist.append((vals[r] - vals[c]) / (r - c))

        # caluclate frequency for each unique value
        freq = 0
        tp = np.unique(vals, return_counts=True)
        for tpx in range(len(tp[0])):
            if (tp[1][tpx] > 1):
                tp1 = tp[1][tpx]
                sev = tp1 * (tp1 - 1) * (2 * tp1 + 5)
                freq = freq + sev

        # Repetition correction based on freq, se = Var(s) after correction
        se = ((valslen * (valslen - 1) * (2 * valslen + 5) - freq) / 18.0) ** 0.5

        # no_of_vals = len(boxlist)

        # calculate K values
        k = stats.norm.ppf(1 - (alpha / 2)) * se

        # calculate slope
        slope = np.median(boxlist)

        # result of senSlope
        senret = {"slope": slope, "k": k, "se": se}

        return senret

    def showRet(self, figure_on=True, num=1000):
        ''' show the Result '''
        # print results
        print("\nMannKendall Test\n")
        for key in self.mkret.keys():
            print(f"{key}: {self.mkret[key]}\n")
        print("\nSen's Slope\n")
        for key in self.senret.keys():
            print(f"{key}: {self.senret[key]}\n")

        # plot
        if figure_on == True:
            if "x" in self.kwargs.keys():
                x = self.kwargs["x"]
            else:
                x = np.arange(len(self.vals))

            fit_line = curve_fit.PolyCurve(x, self.vals, deg=1)
            fig, draw = fit_line.plot(num=num)
            draw.set(title="MannKendall Test", labelx="X", labely="Y", gridy=True, gridx=True)
            fit_line.pnm[1] = self.senret["slope"]
            x_linespace, y_linespace = fit_line.getLinespace(num=num)
            Senslope_plot = draw_plot.PlotDraw(x_linespace, y_linespace, "b--", label="Sen's Slope")
            draw.adddraw(Senslope_plot)
            return fig, draw


if __name__ == '__main__':
    x = np.random.rand(100, )
    mk = MkTest(x).showRet()
