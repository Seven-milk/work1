# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# a function to fitting Univariate distribution
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
import Distribution


class UnivariateDistribution(Distribution.DistributionBase):
    def __init__(self, distribution: stats.rv_continuous):
        ''' init function
        input
            distribution: stats.rv_continuous, the distribution to fit.
        '''
        self.distribution_type = distribution

    def fit(self, data):
        self.distribution = self.distribution_type
        self.params = self.distribution.fit(data)
        self.distribution = self.distribution(*self.params)  # freezing distribution
        self.stats = self.distribution.stats()

    def cdf(self, data):
        return self.distribution.cdf(data)

    def pdf(self, data):
        return self.distribution.pdf(data)

    def ppf(self, q, *args, **kwargs):
        ''' inverse of `cdf` '''
        return self.distribution.ppf(q, *args, **kwargs)

    def plot(self, data, PointNumber=1000):
        ''' plot function, plot PDF & CDF
        input
            PointNumber: int, the point number to plot pdf/cdf line, default=1000
        '''
        dataSort = sorted(data)  # ascending order
        start = dataSort[0] - (dataSort[-1] - dataSort[0]) * 0.1
        stop = dataSort[-1] + (dataSort[-1] - dataSort[0]) * 0.1
        dataSortExtend = np.linspace(start, stop, num=PointNumber)
        bins = int(len(data) / 10)
        # pdf
        plt.figure()
        plt.plot(dataSortExtend, self.distribution.pdf(dataSortExtend), color='b', label='Fit pdf')
        plt.hist(dataSort, density=True, bins=bins, color='r', label='real data')
        plt.xlabel('Data')
        plt.ylabel('Frequency & PDF')
        plt.legend()
        plt.show()
        # cdf
        plt.figure()
        plt.plot(dataSortExtend, self.distribution.cdf(dataSortExtend), color='b', label='Fit cdf')
        plt.hist(dataSort, density=True, cumulative=True, bins=bins, color='r', label='real data')
        plt.xlabel('Data')
        plt.ylabel('CDF')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    data = np.random.normal(0, 1, 1000)
    x = UnivariateDistribution(stats.norm)
    x.fit(data)
    x.plot(data)
