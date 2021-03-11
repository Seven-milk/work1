# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# a function to fitting copula, 初步，需要进一步验证
import numpy as np
import pandas as pd
import Univariatefit
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import copulas
from copulas.univariate import Univariate
from copulas.bivariate import Bivariate
from copulas.multivariate import Multivariate
from copulas.visualization import scatter_3d
from copulas.visualization import scatter_2d


class CopulaDistributionBivariate():
    def __init__(self, cdata_uv, cdistribution: Bivariate):
        ''' init function
        input
            cdata: numpy array (of size n * copula dimension=2) The data(cdf) to fit copula.
            cdistribution: copulas.bivariate, Base class for bivariate copulas., subclass
                copulas.bivariate.[clayton].Clayton
                copulas.bivariate[.frank].Frank
                copulas.bivariate[.gumbel].Gumbel
        '''
        # self.cdata_df = pd.DataFrame(cdata_uv)
        self.cdata = cdata_uv
        self.cdistribution = cdistribution
        self.cdistribution.fit(self.cdata)
        self.data_cpdf = self.cdistribution.pdf(self.cdata)
        self.data_ccdf = self.cdistribution.cumulative_distribution(self.cdata)
        self.parameter = self.cdistribution.to_dict()
        for key in self.parameter.keys():
            print(key, self.parameter[key], '\n')

    def plot(self, PointNumber=100):
        # scatter_2d(self.cdata_df)
        u = np.linspace(0, 1, PointNumber)
        v = np.linspace(0, 1, PointNumber)
        [U, V] = np.meshgrid(u, v)
        c = np.zeros((PointNumber, PointNumber))
        for i in range(PointNumber):
            c[i, :] = self.cdistribution.pdf(np.array([np.full_like(v[:], fill_value=u[i]), v[:]]).T)
        c = (c - c.min()) / (c.max() - c.min())
        # pcolormesh
        plt.figure()
        plt.pcolormesh(U, V, c, cmap='RdBu')
        plt.colorbar(extend='both')
        # data
        plt.scatter(self.cdata[:, 0], self.cdata[:, 1])
        # 3d plot
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(U, V, c, cmap='viridis')


class CopulaDistributionMultivariate():
    def __init__(self, cdata, cdistribution: Multivariate):
        ''' init function
        input
            cdata: numpy array (of size n * copula dimension>=2) The data(cdf) to fit copula.
            cdistribution: copulas.Multivariate, Base class for Multivariate copulas, Abstract class for a multi-variate
            copula object,  subclass
                copulas.multivariate[.gaussian].GaussianMultivariate
                copulas.multivariate[.tree] import Tree, TreeTypes
                copulas.multivariate[.vine] import VineCopula
        '''
        self.cdata = pd.DataFrame(cdata)
        self.cdistribution = cdistribution
        self.cdistribution.fit(cdata)
        self.data_cpdf = self.cdistribution.pdf(self.cdata)
        self.data_ccdf = self.cdistribution.cumulative_distribution(self.cdata)
        self.parameter = self.cdistribution.to_dict()
        for key in self.parameter.keys():
            print(key, self.parameter[key], '\n')

    def plot(self):
        scatter_3d(self.cdata)


if __name__ == "__main__":
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(0, 1, 100)
    u = Univariatefit.UnivariateDistribution(data1, stats.norm)
    v = Univariatefit.UnivariateDistribution(data2, stats.norm)
    cdata = np.array([data1, data1]).T
    cdata_uv = np.array([u.data_cdf, v.data_cdf]).T
    df = pd.DataFrame(cdata)
    c1 = CopulaDistributionBivariate(cdata_uv, copulas.bivariate.Frank())
    c1.plot()
    c2 = CopulaDistributionMultivariate(cdata_uv, copulas.multivariate.GaussianMultivariate())
