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
import useful_func


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

    def fit(self, data, **kwargs):
        '''
        input:
            **kwargs: key word args, it could contains bw_method=None("Scott"), weights=None, reference stats.gaussian_kde
        '''
        self.kde = stats.gaussian_kde(data, **kwargs)
        self._data = np.array(data)
        self._cdf = self.cdf(data)
        self._pdf = self.pdf(data)

    def cdf(self, data):
        if isinstance(data, list) or isinstance(data, np.ndarray):
            cdf = np.array([self.kde.integrate_box_1d(low=-np.inf, high=data[i]) for i in range(len(data))])
        else:
            cdf = self.kde.integrate_box_1d(low=-np.inf, high=data)
        return cdf

    def pdf(self, data):
        return self.kde.pdf(data)

    def ppf(self, percentile):
        if isinstance(percentile, list) or isinstance(percentile, np.ndarray):
            percentile = percentile
        else:
            percentile = [percentile]

        ppf = np.zeros((len(percentile)))
        for i in range(len(percentile)):
            percentile_ = percentile[i]

            # sorted
            cdf_data = np.hstack((self._data.reshape(-1, 1), self._cdf.reshape(-1, 1)))
            cdf_data_sorted = cdf_data[cdf_data[:, 1].argsort()]

            # side
            side = useful_func.side(cdf_data_sorted[:, 1], percentile_)
            left_index = side[0]["index_left"]
            right_index = side[0]["index_right"]
            left_cdf = side[0]["left"]
            right_cdf = side[0]["right"]
            left_data = cdf_data_sorted[left_index, 0]
            right_data = cdf_data_sorted[right_index, 0]

            # intersection
            line1 = [left_data, left_cdf, right_data, right_cdf]
            line2 = [left_data, percentile_, right_data, percentile_]
            ret = useful_func.intersection(line1, line2)
            ppf[i] = ret[0]

        return ppf


class Gringorten(NonparamBase):
    ''' Gringorten nonparametric distribution '''

    def fit(self, data):
        self._data = np.array(data)
        series_ = pd.Series(data)
        cdf = [(series_.rank(axis=0, method="min", ascending=True)[i] - 0.44) / (len(series_) + 0.12) for i in
               range(len(series_))]
        self._cdf = np.array(cdf)

    def cdf(self, data):
        if isinstance(data, list) or isinstance(data, np.ndarray):
            data = data
        else:
            data = [data]

        cdf = np.zeros((len(data)))
        for i in range(len(data)):
            data_ = data[i]
            series_ = pd.Series(np.append(self._data, data_))
            rank = series_.rank(axis=0, method="min", ascending=True).to_numpy()[-1]
            cdf[i] = (rank - 0.44) / (len(self._data) + 0.12)

        return cdf

    def pdf(self, data):
        return None

    def ppf(self, percentile):
        if isinstance(percentile, list) or isinstance(percentile, np.ndarray):
            percentile = percentile
        else:
            percentile = [percentile]

        ppf = np.zeros((len(percentile)))
        for i in range(len(percentile)):
            percentile_ = percentile[i]

            # sorted
            cdf_data = np.hstack((self._data.reshape(-1, 1), self._cdf.reshape(-1, 1)))
            cdf_data_sorted = cdf_data[cdf_data[:, 1].argsort()]

            # side
            side = useful_func.side(cdf_data_sorted[:, 1], percentile_)
            left_index = side[0]["index_left"]
            right_index = side[0]["index_right"]
            left_cdf = side[0]["left"]
            right_cdf = side[0]["right"]
            left_data = cdf_data_sorted[left_index, 0]
            right_data = cdf_data_sorted[right_index, 0]

            # intersection
            line1 = [left_data, left_cdf, right_data, right_cdf]
            line2 = [left_data, percentile_, right_data, percentile_]
            ret = useful_func.intersection(line1, line2)
            ppf[i] = ret[0]

        return ppf


class EmpiricalDistribution(NonparamBase):
    ''' Empirical Distribution '''

    def fit(self, data):
        # /(len(data) + 1) to avoid max(cdf) == 1
        self._data = np.array(data)
        cdf = [len([x_ for x_ in data if x_ <= x]) / (len(data) + 1) for x in data]
        self._cdf = np.array(cdf)

    def cdf(self, data):
        if isinstance(data, list) or isinstance(data, np.ndarray):
            data = data
        else:
            data = [data]

        cdf = np.zeros((len(data)))
        for i in range(len(data)):
            data_ = data[i]
            cdf[i] = len([x_ for x_ in self._data if x_ <= data_]) / (len(self._data) + 1)

        return cdf

    def pdf(self, data):
        return None

    def ppf(self, percentile):
        if isinstance(percentile, list) or isinstance(percentile, np.ndarray):
            percentile = percentile
        else:
            percentile = [percentile]

        ppf = np.zeros((len(percentile)))
        for i in range(len(percentile)):
            percentile_ = percentile[i]

            # sorted
            cdf_data = np.hstack((self._data.reshape(-1, 1), self._cdf.reshape(-1, 1)))
            cdf_data_sorted = cdf_data[cdf_data[:, 1].argsort()]

            # side
            side = useful_func.side(cdf_data_sorted[:, 1], percentile_)
            left_index = side[0]["index_left"]
            right_index = side[0]["index_right"]
            left_cdf = side[0]["left"]
            right_cdf = side[0]["right"]
            left_data = cdf_data_sorted[left_index, 0]
            right_data = cdf_data_sorted[right_index, 0]

            # intersection
            line1 = [left_data, left_cdf, right_data, right_cdf]
            line2 = [left_data, percentile_, right_data, percentile_]
            ret = useful_func.intersection(line1, line2)
            ppf[i] = ret[0]

        return ppf


if __name__ == '__main__':
    # general set
    np.random.seed(15)
    # x = np.random.rand(100, )
    x = np.random.normal(0, 1, 1000)

    # kde
    kde = KdeDistribution()
    kde.fit(x)
    kdecdf = kde.cdf(x)
    print("kde ppf 0.5", kde.ppf(0.5))

    # normal
    normal = Univariatefit.UnivariateDistribution(stats.norm)
    normal.fit(x)
    normal_cdf = normal.cdf(x)
    print("norm ppf 0.5", normal.ppf(0.5))

    # gringorten
    gg = Gringorten()
    gg.fit(x)
    ggcdf = gg.cdf(x)
    print("gg ppf 0.5", gg.ppf(0.5))

    # Empirical
    ed = EmpiricalDistribution()
    ed.fit(x)
    edcdf = ed.cdf(x)
    print("ed ppf 0.5", ed.ppf(0.5))

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