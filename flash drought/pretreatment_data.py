# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# pretreatment data and save
import numpy as np
import pandas as pd
import os
import Workflow
import Nonparamfit, Univariatefit, Distribution
import draw_plot
import Evaluation
from scipy import stats
from scipy.stats import pearsonr


class CombineNoahSm(Workflow.WorkBase):
    ''' Work, Combine SoilMoi0_10cm_inst/10_40/40_100 into SoilMoi0_100cm '''

    def __init__(self, home, save=False):
        self.Sm_path = [os.path.join(home, sm) for sm in
                        ['SoilMoi0_10cm_inst_19480101.0300_20141231.2100.npy',
                         'SoilMoi10_40cm_inst_19480101.0300_20141231.2100.npy',
                         'SoilMoi40_100cm_inst_19480101.0300_20141231.2100.npy']]
        self.save = save
        self._info = 'Noah Sm Combination 0-100cm'

    def __call__(self):
        print("start combine")
        print("load data")
        Sm0_10 = np.load(self.Sm_path[0], mmap_mode='r')
        Sm10_40 = np.load(self.Sm_path[1], mmap_mode='r')
        Sm40_100 = np.load(self.Sm_path[2], mmap_mode='r')
        print("load complete")
        print("sum")
        self.Sm0_100 = np.zeros_like(Sm0_10)
        self.Sm0_100[:, 0] = Sm0_10[:, 0]
        self.Sm0_100[:, 1:] = Sm0_10[:, 1:] + Sm10_40[:, 1:] + Sm40_100[:, 1:]
        print("sum complete")
        print("save")
        if self.save == True:
            np.save('SoilMoi0_100cm_inst_19480101.0300_20141231.2100.npy', self.Sm0_100)
        print("complete!")

    def __repr__(self):
        return f"This is CombineNoahSm, info: {self._info}, Combine SoilMoi0_10cm_inst/10_40/40_100 into SoilMoi0_100cm"

    def __str__(self):
        return f"This is CombineNoahSm, info: {self._info}, Combine SoilMoi0_10cm_inst/10_40/40_100 into SoilMoi0_100cm"


class UpscaleTime(Workflow.WorkBase):
    ''' Work, Upscale time series, such as, upscale 3H series to daily series (average or sum), upscale daily series to
        pentad series(average or sum)
    '''

    def __init__(self, original_series, multiple: int, up_method: callable = lambda x: sum(x)/len(x),
                 original_date=None, save_path=None, combine=True, info=""):
        ''' init function
        input:
            up_method: upscale method, default = lambda x: sum(x)/len(x) = mean, note 1 0 1 -> 2/3 is not 2/2
            original_series: 1D or 2D np.array, original series to upscale, when daily_series is a 2D array,
                            m(time) * n(other, such as grid points)
            original_date: 1D array like, original date corresponding to original series
            save_path: str, path to save, default=None, namely not save
            combine: whether combine upscale_date & upscale_series to output and save
            multiple: int, upscale time from original_series to objective_series
                D -> pentad: 5
                3H -> D: 8
                D -> Y: 365
            info: str, informatiom for this Class to print, shouldn't too long

        output:
            {self.save_path}_date.npy, {self.save_path}_series.npy: upscale date and series
            {self.save_path}.npy: combine output, the first col is upscale_date
        '''

        self.original_series = original_series
        self.up_method = up_method

        if isinstance(original_date, list) == True or isinstance(original_date, np.ndarray) == True:
            self.original_date = original_date
        else:
            self.original_date = np.arange(len(self.original_series))

        self.save_path = save_path
        self.multiple = multiple
        self._info = info
        self.combine = combine

    def __call__(self):
        ''' implement WorkBase.__call__ '''
        print("start upScale")
        print("start calculate")
        upscale_date, upscale_series = self.upScale()
        print("complete calculate")

        # whether combine and save
        if self.combine == True:
            upscale_series = np.hstack((upscale_date.reshape(len(upscale_date), 1), upscale_series))
            if self.save_path != None:
                np.save(self.save_path, upscale_series)

            print("complete upScale")
            return upscale_series

        else:
            if self.save_path != None:
                np.save(self.save_path + '_date', upscale_date)
                np.save(self.save_path + '_series', upscale_series)

            print("complete upScale")
            return upscale_date, upscale_series

    def upScale(self):
        ''' up scale series '''

        # cal the series num which can be contain in cal period
        multiple = self.multiple
        up_method = self.up_method
        num_in = len(self.original_series) // multiple
        num_out = len(self.original_series) - num_in * multiple

        # del [-numout:] to make sure len(original_series) can be exact division by self.multiple
        if len(self.original_series.shape) > 1:
            original_series = self.original_series[:-num_out, :] if num_out != 0 else self.original_series
            upscale_series = np.zeros((num_in, self.original_series.shape[1]), dtype="float")
        else:
            original_series = self.original_series[:-num_out] if num_out != 0 else self.original_series
            upscale_series = np.zeros((num_in, ), dtype="float")

        original_date = self.original_date[:-num_out] if num_out != 0 else self.original_date
        upscale_date = np.zeros((num_in,), dtype="float")

        # cal upscale_series & upscale_date
        for i in range(num_in):
            if len(self.original_series.shape) > 1:
                upscale_series[i, :] = up_method(original_series[i * multiple: (i + 1) * multiple, :])  # axis=0
            else:
                upscale_series[i] = up_method(original_series[i * multiple: (i + 1) * multiple])

            # center date(depend on multiple, odd is center, even is the right of center)
            upscale_date[i] = original_date[i * multiple + multiple // 2]

        return upscale_date, upscale_series

    def __repr__(self):
        return f"This is UpscaleTime, info: {self._info}, Upscale time series"

    def __str__(self):
        return f"This is UpscaleTime, info: {self._info}, Upscale time series"


class CalSmPercentile(Workflow.WorkBase):
    ''' Work, calculate SmPercentile series from SM series based on a given distribution
    the sms with same month put together to build distribution
    '''

    def __init__(self, sm, date, format, distribution: Distribution.DistributionBase, combine=True, save_path=None, info=""):
        ''' init function
        input:
            sm: 1D or 2D array like, soil moisture series, m(time) * n(other, such as grid points)
            date: 1D array, date will further change into pd.DatetimeIndex
            format: format to change date, e.g. 19980101 '%Y%m%d'
            distribution: a instance having the func cdf to cal cdf
            save_path: str, home path to save, if save_path=None(default), do not save
            combine: whether combine date & sm_percentile to output and save
            info: str, informatiom for this Class to print and save in save_path, shouldn't too long

        output:
            sm_percentile: sm percentile(combine date or not)
            save_path.npy: sm_percentile file

        '''
        self._sm = sm
        self.date = date
        self.format = format
        self.save_path = save_path
        self.distribution = distribution
        self.combine = combine
        self._info = info

    def __call__(self):
        ''' implement WorkBase.__call__ '''
        sm_percentile = np.zeros_like(self._sm, dtype="float")
        date = pd.to_datetime(self.date, format=self.format)

        if len(self._sm.shape) > 1:
            print(f"all series number {self._sm.shape[1]}")
            # each series
            for i in range(self._sm.shape[1]):
                sm_ = self._sm[:, i]
                # each month
                for j in range(12):
                    index_month = [date_ for date_ in range(len(sm_)) if date[date_].month == j + 1]
                    sm_month = sm_[index_month]
                    self.distribution.fit(sm_month)
                    sm_month_percentile = self.distribution.cdf(sm_month)
                    sm_percentile[index_month, i] = sm_month_percentile

                print(f"sm series {i} calculated completely")

        else:
            sm_ = self._sm
            for j in range(12):
                index_month = [date_ for date_ in range(len(sm_)) if date[date_].month == j + 1]
                sm_month = sm_[index_month]
                self.distribution.fit(sm_month)
                sm_month_percentile = self.distribution.cdf(sm_month)
                sm_percentile[index_month] = sm_month_percentile

        # combine
        if self.combine == True:
            sm_percentile_ = sm_percentile if len(self._sm.shape) > 1 else sm_percentile.reshape(len(sm_percentile), 1)
            sm_percentile = np.hstack((self.date.reshape(len(self.date), 1), sm_percentile_))

        # save result
        if self.save_path != None:
            np.save(self.save_path, sm_percentile)

        return sm_percentile

    def __repr__(self):
        return f"This is CalSmPercentile, info: {self._info}, calculate SmPercentile series from SM series"

    def __str__(self):
        return f"This is CalSmPercentile, info: {self._info}, calculate SmPercentile series from SM series"


class CalSmPercentileMultiDistribution(CalSmPercentile):
    ''' Work, calculate SmPercentile series from SM series based on multiple distribution '''

    def __init__(self, sm, date, format, distribution: list, nonparamdistribution: Nonparamfit.NonparamBase, alpha=0.05,
                 combine=True, save_path=None, info=""):
        ''' init function: similar with CalSmPercentile
            note:
                distribution: list of multiple-distributions

        '''
        self._sm = sm
        self.date = date
        self.format = format
        self.save_path = save_path
        self.distribution = distribution
        self.combine = combine
        self._info = info
        self.nonparamdistribution = nonparamdistribution
        self.alpha = alpha

    def __call__(self):
        ''' implement WorkBase.__call__ '''
        sm_percentile = np.zeros_like(self._sm, dtype="float")
        date = pd.to_datetime(self.date, format=self.format)

        if len(self._sm.shape) > 1:
            distribution_ret = np.zeros((12, self._sm.shape[1]), dtype=int)
            print(f"all series number {self._sm.shape[1]}")
            # each series
            for i in range(self._sm.shape[1]):
                sm_ = self._sm[:, i]
                # each month
                for j in range(12):
                    index_month = [date_ for date_ in range(len(sm_)) if date[date_].month == j + 1]
                    sm_month = sm_[index_month]

                    # evaluation distribution
                    evaluation_ret = pd.DataFrame(np.zeros((len(self.distribution), 2)), columns=['kstest', 'aic'],
                                                  index=[i for i in range(len(self.distribution))], dtype=float)
                    evaluation = Evaluation.Evaluation()
                    for k in range(len(self.distribution)):
                        self.distribution[k].fit(sm_month)
                        kstest = evaluation.kstest(sm_month, self.distribution[k].cdf)
                        kstest = kstest[1] > 1 - self.alpha  # p_value > 1-alpha, if true, passed
                        aic = evaluation.aic(sm_month, self.distribution[k].ppf, len(self.distribution[k].params))
                        evaluation_ret.iloc[k, 0] = kstest
                        evaluation_ret.iloc[k, 1] = aic

                    # select distribution
                    evaluation_ret = evaluation_ret[evaluation_ret["kstest"]==1]
                    if len(evaluation_ret) > 1:
                        index_distribution = int(evaluation_ret.index[evaluation_ret.aic.argmin()])
                        distribution_select = self.distribution[index_distribution]
                    else:
                        index_distribution = -1
                        distribution_select = self.nonparamdistribution

                    # save in distribution_ret
                    distribution_ret[j, i] = index_distribution

                    # fit and cal sm_month_percentile
                    distribution_select.fit(sm_month)
                    sm_month_percentile = distribution_select.cdf(sm_month)
                    sm_percentile[index_month, i] = sm_month_percentile

                print(f"sm series {i} calculated completely")

            distribution_ret = pd.DataFrame(distribution_ret, index=[f"month{i_ + 1}" for i_ in range(12)],
                                            columns=[i_ for i_ in range(self._sm.shape[1])])

        else:
            distribution_ret = np.zeros((12, ), dtype=int)
            sm_ = self._sm
            for j in range(12):
                index_month = [date_ for date_ in range(len(sm_)) if date[date_].month == j + 1]
                sm_month = sm_[index_month]

                # evaluation distribution
                evaluation_ret = pd.DataFrame(np.zeros((len(self.distribution), 2)), columns=['kstest', 'aic'],
                                              index=[i for i in range(len(self.distribution))], dtype=float)
                evaluation = Evaluation.Evaluation()

                for i in range(len(self.distribution)):
                    self.distribution[i].fit(sm_month)
                    kstest = evaluation.kstest(sm_month, self.distribution[i].cdf)
                    kstest = kstest[1] > 1 - self.alpha  # p_value > 1-alpha, if true, passed
                    aic = evaluation.aic(sm_month, self.distribution[i].ppf, len(self.distribution[i].params))
                    evaluation_ret.iloc[i, 0] = kstest
                    evaluation_ret.iloc[i, 1] = aic

                # select distribution
                evaluation_ret = evaluation_ret[evaluation_ret["kstest"] == 1]
                if len(evaluation_ret) > 1:
                    index_distribution = int(evaluation_ret.index[evaluation_ret.aic.argmin()])
                    distribution_select = self.distribution[index_distribution]
                else:
                    index_distribution = -1
                    distribution_select = self.nonparamdistribution

                # save in distribution_ret
                distribution_ret[j] = index_distribution

                # fit and cal sm_month_percentile
                distribution_select.fit(sm_month)
                sm_month_percentile = distribution_select.cdf(sm_month)
                sm_percentile[index_month] = sm_month_percentile

            distribution_ret = pd.DataFrame(distribution_ret, index=[f"month{i_ + 1}" for i_ in range(12)])

        # combine
        if self.combine == True:
            sm_percentile_ = sm_percentile if len(self._sm.shape) > 1 else sm_percentile.reshape(len(sm_percentile), 1)
            sm_percentile = np.hstack((self.date.reshape(len(self.date), 1), sm_percentile_))

        # save result
        if self.save_path != None:
            np.save(self.save_path, sm_percentile)
            distribution_ret.to_excel(self.save_path + "_distribution_ret.xlsx")

        return sm_percentile, distribution_ret


class CompareSmPercentile(Workflow.WorkBase):
    ''' Work, compare sm_rz_pentad and sm_percentile_rz_pentad: differences result from the fit and section
     calculation '''

    def __init__(self, sm, sm_percentile, date, info=""):
        ''' init function
        input:
            sm & sm_percentile: 1D array, sm and sm percentile
            date: pd.DatetimeIndex, the models' date
        '''

        self.sm = sm
        self.sm_percentile = sm_percentile
        self._info = info
        self.date = date

    def __call__(self):
        """ implement WorkBase.__call__ """
        # plot set
        f = draw_plot.Figure()
        d_sm = draw_plot.Draw(f.ax, f, gridy=True, labelx="Date", labely="Sm / m", legend_on=True)
        d_sm_percentile = draw_plot.Draw(f.ax.twinx(), f, gridy=True, labely="Sm percentile", legend_on=True)

        # add plot
        d_sm.adddraw(draw_plot.PlotDraw(self.date, self.sm, "b.", alpha=0.5, markersize=0.5, label="sm"))
        d_sm_percentile.adddraw(draw_plot.PlotDraw(self.date, self.sm_percentile, "r.", markersize=0.5, label="sm_percentile"))

        # show
        f.show()

        # coefficient, p_value < alpha, passed
        r, p_value = pearsonr(self.sm, self.sm_percentile)
        print(f"r={r}, p_value={p_value}")

    def __repr__(self):
        return f"This is CompareSmPercentile, info: {self._info}, compare sm and sm_percentile"

    def __str__(self):
        return f"This is CompareSmPercentile, info: {self._info}, compare sm and sm_percentile"


def combine_Noah_SM():
    # combine Noah Sm, sum
    cns = CombineNoahSm(home='H:/research/flash_drough/GLDAS_Noah', save=True)
    cns()


def Upscale_Noah_D():
    # Upscale Noah from 3H to D
    original_series = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101.0300_20141231.2100.npy']
    save_path = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101_20141231_D',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_D']

    for i in range(len(original_series)):
        original_series_ = np.load(original_series[i], mmap_mode='r')
        original_series_post = original_series_[:7, :]  # start from 0300, the first day contains 7 days rather than 8 days
        original_series_after = original_series_[7:, :]

        D_post = UpscaleTime(original_series=original_series_post[:, 1:], multiple=7,
                                original_date=original_series_post[:, 0], save_path=None,
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:])()
        D_after = UpscaleTime(original_series=original_series_after[:, 1:], multiple=8,
                                original_date=original_series_after[:, 0], save_path=None,
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:])()

        D_ = np.vstack((D_post, D_after))
        np.save(save_path[i], D_)


def Upscale_Noah_Pentad():
    # Upscale Noah from D to Pentad
    original_series = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101_20141231_D.npy',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_D.npy']
    save_path = ['H:/research/flash_drough/GLDAS_Noah/RootMoist_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_10cm_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi10_40cm_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi40_100cm_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad']
    upscale_Noah = Workflow.WorkFlow()

    for i in range(len(original_series)):
        original_series_ = np.load(original_series[i], mmap_mode='r')
        upscale_Noah_ = UpscaleTime(original_series=original_series_[:, 1:], multiple=5,
                                original_date=original_series_[:, 0], save_path=save_path[i],
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:])
        upscale_Noah += upscale_Noah_

    upscale_Noah()


def Upscale_CLS_Pentad():
    original_series = 'H:/research/flash_drough/GLDAS_Catchment/SoilMoist_RZ_tavg_19480101_20141230.npy'
    save_path = 'H:/research/flash_drough/GLDAS_Catchment/SoilMoist_RZ_tavg_19480101_20141230_Pentad'
    original_series = np.load(original_series, mmap_mode='r')
    upscale_CLS = UpscaleTime(original_series=original_series[:, 1:], multiple=5, original_date=original_series[:, 0],
                              save_path=save_path, combine=True, info=save_path[save_path.rfind("/") + 1:])
    upscale_CLS()


def smpercentile_Noah_0_100cm_kde():
    sm_ = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad.npy', mmap_mode="r")
    sm = sm_[:, 1:]
    date = sm_[:, 0]
    format = '%Y%m%d'

    # use kde
    kde = Nonparamfit.KdeDistribution()

    # cal sm percentile
    csp_kde = CalSmPercentile(sm, date, format=format, distribution=kde, info="Kde distribution sm percentile",
                              save_path="SoilMoi0_100cm_inst_19480101_20141231_Pentad_KdeSmPercentile")
    csp_kde()


def smpercentile_Noah_0_100cm_multiple_distribution():
    sm_ = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad.npy', mmap_mode="r")
    sm = sm_[:, 1:]
    date = sm_[:, 0]
    format = '%Y%m%d'

    # nonparam
    nonparamdistribution = Nonparamfit.Gringorten()

    # distributions
    distribution = [Univariatefit.UnivariateDistribution(stats.expon), Univariatefit.UnivariateDistribution(stats.gamma),
                    Univariatefit.UnivariateDistribution(stats.beta), Univariatefit.UnivariateDistribution(stats.lognorm),
                    Univariatefit.UnivariateDistribution(stats.logistic), Univariatefit.UnivariateDistribution(stats.pareto),
                    Univariatefit.UnivariateDistribution(stats.weibull_min), Univariatefit.UnivariateDistribution(stats.genextreme)]

    # cal sm percentile
    cspmd = CalSmPercentileMultiDistribution(sm, date, format, distribution=distribution,
                                             nonparamdistribution=nonparamdistribution,
                                             info="multiple distribution sm percentile",
                                             save_path="SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile")
    cspmd()


def compareSmSmPercentile_Noah_0_100cm_kde_multiple_dis():
    sm_ = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad.npy', mmap_mode="r")
    sm_percentile_kde_ = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad_KdeSmPer'
                             'centile.npy', mmap_mode="r")
    sm_percentile_multi_dis_ = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis'
                                      '_SmPercentile.npy', mmap_mode="r")

    sm = sm_[:, 1:]
    sm_percentile_kde = sm_percentile_kde_[:, 1:]
    sm_percentile_multi_dis = sm_percentile_multi_dis_[:, 1:]

    date = pd.to_datetime(sm_[:, 0], format='%Y%m%d')

    point = 117

    csp_kde = CompareSmPercentile(sm[:, point], sm_percentile_kde[:, point], date)
    csp_multi_dis = CompareSmPercentile(sm[:, point], sm_percentile_multi_dis[:, point], date)
    csp_kde()
    csp_multi_dis()

    csp_c = CompareSmPercentile(sm_percentile_multi_dis[:, point], sm_percentile_kde[:, point], date)
    csp_c()

    # diff
    diff = sm_percentile_kde - sm_percentile_multi_dis
    diff_mean = abs(diff).mean(axis=0)

    mean_ = (sm_percentile_kde + sm_percentile_multi_dis) / 2
    diff_relative = abs(diff / mean_) * 100
    diff_relative_mean = diff_relative.mean(axis=0)

    diff_ = np.vstack((diff_mean, diff_relative_mean))
    diff = pd.DataFrame(diff_, index=["diff_mean", "diff_relative_mean"], columns=list(range(sm_percentile_kde.shape[1])))
    diff.to_excel("diff.xlsx")


def smpercentile_Noah_0_100cm_multiple_distribution_Region_mean():
    sm_ = np.load('H:/research/flash_drough/GLDAS_Noah/SoilMoi0_100cm_inst_19480101_20141231_Pentad.npy', mmap_mode="r")
    sm = sm_[:, 1:].mean(axis=1)
    date = sm_[:, 0]
    format = '%Y%m%d'

    # nonparam
    nonparamdistribution = Nonparamfit.Gringorten()

    # distributions
    distribution = [Univariatefit.UnivariateDistribution(stats.expon), Univariatefit.UnivariateDistribution(stats.gamma),
                    Univariatefit.UnivariateDistribution(stats.beta), Univariatefit.UnivariateDistribution(stats.lognorm),
                    Univariatefit.UnivariateDistribution(stats.logistic), Univariatefit.UnivariateDistribution(stats.pareto),
                    Univariatefit.UnivariateDistribution(stats.weibull_min), Univariatefit.UnivariateDistribution(stats.genextreme)]

    # cal sm percentile
    cspmd = CalSmPercentileMultiDistribution(sm, date, format, distribution=distribution,
                                             nonparamdistribution=nonparamdistribution,
                                             info="multiple distribution sm percentile from region mean sm",
                                             save_path="SoilMoi0_100cm_inst_19480101_20141231_Pentad_muldis_SmPercentile_RegionMean")
    cspmd()


def Upscale_Noah_mutileVariables_Pentad():
    # Upscale Noah from 3H to Pentad
    original_series = ['H:/research/flash_drough/GLDAS_Noah/CanopInt_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/AvgSurfT_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/Wind_f_inst_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/ECanop_tavg_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/PotEvap_tavg_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/ESoil_tavg_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/Rainf_f_tavg_19480101.0300_20141231.2100.npy',
                       'H:/research/flash_drough/GLDAS_Noah/Tair_f_inst_19480101.0300_20141231.2100.npy']
    save_path = ['H:/research/flash_drough/GLDAS_Noah/CanopInt_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/AvgSurfT_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/Wind_f_inst_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/ECanop_tavg_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/PotEvap_tavg_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/ESoil_tavg_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/Rainf_f_tavg_19480101_20141231_Pentad',
                 'H:/research/flash_drough/GLDAS_Noah/Tair_f_inst_19480101_20141231_Pentad']

    for i in range(len(original_series)):
        original_series_ = np.load(original_series[i], mmap_mode='r')
        original_series_post = original_series_[:39, :]  # start from 0300, the first day contains 39 3Hours rather than 40 3Hours
        original_series_after = original_series_[39:, :]

        # up_method: all average()
        D_post = UpscaleTime(original_series=original_series_post[:, 1:], multiple=39,
                                original_date=original_series_post[:, 0], save_path=None,
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:])()
        D_after = UpscaleTime(original_series=original_series_after[:, 1:], multiple=40,
                                original_date=original_series_after[:, 0], save_path=None,
                                combine=True, info=save_path[i][save_path[i].rfind("/") + 1:])()

        D_ = np.vstack((D_post, D_after))
        np.save(save_path[i], D_)


if __name__ == '__main__':
    # # combine Noah SM between different layers
    # combine_Noah_SM()

    # # upscale noah and CLS sm
    # Upscale_Noah_D()
    # Upscale_Noah_Pentad()
    # Upscale_CLS_Pentad()

    # cal sm percentile
    # smpercentile_Noah_0_100cm_kde()
    # smpercentile_Noah_0_100cm_multiple_distribution()

    # compare sm with sm percentile
    # compareSmSmPercentile_Noah_0_100cm_kde_multiple_dis()

    # cal sm percentile RegionMean
    # smpercentile_Noah_0_100cm_multiple_distribution_Region_mean()

    # upscale noah multiple variables
    Upscale_Noah_mutileVariables_Pentad()