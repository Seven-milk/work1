# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# a function to fitting copula
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from Univariatefit import UnivariateDistribution


class CopulaDistribution():
    def __init__(self, cdata, u: UnivariateDistribution, v: UnivariateDistribution, cdistribution: Copula,
                 method='cmle'):
        ''' init function
        input
            cdata: numpy array (of size n * copula dimension) The data to fit copula.
            u: stats.rv_continuous, marginal distribution
            v: stats.rv_continuous, marginal distribution
            cdistribution: Copula, the copula distribution to fit, subclass:
                ArchimedeanCopula(family='Clayton', 'Gumbel', 'Joe', 'Frank', 'Ali-Mikhail-Haq')
                GaussianCopula
                StudentCopula
        '''
        self.cdata = cdata
        self.u = u
        self.v = v
        self.cdistribution = cdistribution
        self.params, self.estimationData = self.cdistribution.fit(self.cdata, method=method, verbose=True)
        self.data_cdf = [self.cdistribution.cdf([self.u.data_cdf[i], self.v.data_cdf[i]]) for i in
                         range(len(self.u.data_cdf))]
        self.data_pdf = [self.cdistribution.pdf([self.u.data_cdf[i], self.v.data_cdf[i]]) for i in
                         range(len(self.u.data_cdf))]
        self.dim = self.cdistribution.dim
        if self.cdistribution.dim == 2:
            self.data_correlation = self.cdistribution.correlations(self.cdata)

    def plot2D(self):
        U, V, cdf = cdf_2d(self.cdistribution)
        U, V, pdf = pdf_2d(self.cdistribution)
        fig = plt.figure()
        # cdf
        ax1 = fig.add_subplot(121, projection='3d', title="Copula CDF")
        U_mesh, V_mesh = np.meshgrid(U, V)
        ax1.plot_surface(U_mesh, V_mesh, cdf, cmap=cm.Blues)
        ax1.plot_wireframe(U_mesh, V_mesh, cdf, color='black', alpha=0.3)
        # pdf
        ax2 = fig.add_subplot(122, title="Copula PDF")
        ax2.contour(U_mesh, V_mesh, pdf, levels=np.arange(0, 5, 0.15), zorder=20)
        # real data
        z = [self.cdistribution.pdf([self.u.data_cdf[i], self.v.data_cdf[i]]) for i in range(len(self.u.data_cdf))]
        ax2.scatter(self.u.data_cdf, self.v.data_cdf, s=3, label="real data", cmap='YlOrRd', zorder=30)  # c=np.log10(z),
        # show
        plt.legend()
        plt.show()

    # def plot3D(self):


if __name__ == "__main__":
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(0, 1, 100)
    u = UnivariateDistribution(data1, stats.norm)
    v = UnivariateDistribution(data2, stats.norm)
    cdata = np.array([data1, data1]).T
    c = CopulaDistribution(cdata, u, v, GaussianCopula(dim=2))  # ArchimedeanCopula(family='gumbel', dim=2)
    c.plot2D()
