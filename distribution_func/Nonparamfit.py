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


class NonparamBase(abc.ABC):
    ''' Nonparametric distribution Base abstract class '''


class KdeDistribution(NonparamBase):
    ''' Kde dirtribution class '''

    def cdf(self, data, **kwargs):
        '''
        input:
            **kwargs: key word args, it could contains bw_method=None, weights=None, reference stats.gaussian_kde
        '''
        kde = stats.gaussian_kde(data, **kwargs)
        cdf = np.array([kde.integrate_box_1d(low=0, high=data[i]) for i in range(len(data))])
        return cdf


class Gringorten(NonparamBase):
    ''' Gringorten nonparametric distribution '''

    def cdf(self, data):
        series_ = pd.Series(data)
        cdf = [(series_.rank(axis=0, method="min", ascending=True)[i] - 0.44) / (len(series_) + 0.12) for
                         i in range(len(series_))]
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
    normal = Univariatefit.UnivariateDistribution(x, stats.norm)
    normal_cdf = normal.data_cdf

    # gringorten
    gg = Gringorten()
    ggcdf = gg.cdf(x)

    # combine
    ret = np.vstack((x, kdecdf, ggcdf, normal_cdf))
    ret = ret.T[np.lexsort(ret[::-1, :])].T

    # print
    for i in zip(x, ggcdf, kdecdf):
        print(*i)

    print(max(kdecdf))

    f = draw_plot.Figure()
    draw_x = draw_plot.Draw(f.ax, f, gridy=True, labelx="x", labely="cdf_x", legend_on=True)

    draw_cdf = draw_plot.Draw(f.ax.twinx(), f, gridy=True, labelx="x", labely="cdf", legend_on=True)

    draw_x.adddraw(draw_plot.HistDraw(ret[0, :], label="x", cumulative=True, density=True))
    draw_cdf.adddraw(draw_plot.PlotDraw(ret[0, :], ret[1, :], "r", label="kde"))
    draw_cdf.adddraw(draw_plot.PlotDraw(ret[0, :], ret[2, :], "g", label="gg"))
    draw_cdf.adddraw(draw_plot.PlotDraw(ret[0, :], ret[3, :], "k", label="normal"))

    f.show()