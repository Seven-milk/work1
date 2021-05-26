# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Nonparametric distribution
import abc
import pandas as pd
import numpy as np
from scipy import stats
import draw_plot
import Univariatefit
import Distribution


class NonparamBase(Distribution.DistributionBase):
    ''' Nonparametric distribution Base abstract class '''
    def fit(self, data):
        pass

    def cdf(self, data):
        pass

    def pdf(self, data):
        pass


class KdeDistribution(NonparamBase):
    ''' Kde dirtribution class '''

    def fit(self, data):
        pass

    def cdf(self, data, **kwargs):
        '''
        input:
            **kwargs: key word args, it could contains bw_method=None("Scott"), weights=None, reference stats.gaussian_kde
        '''
        kde = stats.gaussian_kde(data, **kwargs)
        cdf = np.array([kde.integrate_box_1d(low=-np.inf, high=data[i]) for i in range(len(data))])
        return cdf


class Gringorten(NonparamBase):
    ''' Gringorten nonparametric distribution '''

    def fit(self, data):
        pass

    def cdf(self, data):
        series_ = pd.Series(data)
        cdf = [(series_.rank(axis=0, method="min", ascending=True)[i] - 0.44) / (len(series_) + 0.12) for
                         i in range(len(series_))]
        cdf = np.array(cdf)
        return cdf


class EmpiricalDistribution(NonparamBase):
    ''' Empirical Distribution '''

    def fit(self, data):
        pass

    def cdf(self, data):
        # /(len(data) + 1) to avoid max(cdf) == 1
        cdf = [len([x_ for x_ in data if x_ <= x]) / (len(data) + 1) for x in data]
        cdf = np.array(cdf)
        return cdf


if __name__ == '__main__':
    # general set
    np.random.seed(15)
    # x = np.random.rand(100, )
    x = np.random.normal(0, 1, 1000)

    # kde
    kde = KdeDistribution()
    kdecdf = kde.cdf(x)

    # normal
    normal = Univariatefit.UnivariateDistribution(stats.norm)
    normal.fit(x)
    normal_cdf = normal.cdf(x)

    # gringorten
    gg = Gringorten()
    ggcdf = gg.cdf(x)

    # Empirical
    ed = EmpiricalDistribution()
    edcdf = ed.cdf(x)

    # combine
    ret = np.vstack((x, kdecdf, ggcdf, normal_cdf, edcdf))
    ret = ret.T[np.lexsort(ret[::-1, :])].T

    # print
    for i in range(len(ret)):
        print(ret[:, i])

    print(max(kdecdf))

    f = draw_plot.Figure()
    draw_x = draw_plot.Draw(f.ax, f, gridy=True, labelx="x", labely="cdf_x", legend_on=True)

    draw_cdf = draw_plot.Draw(f.ax.twinx(), f, gridy=True, labelx="x", labely="cdf", legend_on=True)

    linewidth = 1
    draw_x.adddraw(draw_plot.HistDraw(ret[0, :], label="x", cumulative=True, density=True, bins=100))
    draw_cdf.adddraw(draw_plot.PlotDraw(ret[0, :], ret[1, :], "r", label="kde", linewidth=linewidth))
    draw_cdf.adddraw(draw_plot.PlotDraw(ret[0, :], ret[2, :], "g", label="gg", linewidth=linewidth))
    draw_cdf.adddraw(draw_plot.PlotDraw(ret[0, :], ret[3, :], "k", label="normal", linewidth=linewidth))
    draw_cdf.adddraw(draw_plot.PlotDraw(ret[0, :], ret[4, :], "y", label="empirical", linewidth=linewidth))

    f.show()