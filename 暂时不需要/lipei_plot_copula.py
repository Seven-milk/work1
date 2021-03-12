# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 沛总画图
import os
import pandas as pd
import numpy as np
from Univariatefit import UnivariateDistribution
from scipy import stats
from Copulafit import CopulaDistributionBivariate
import copulas
from copulas.bivariate import Bivariate

home = 'H:/工作/李沛'
data1 = pd.read_excel(os.path.join(home, '东三省亩产.xlsx'), '辽宁')
data2 = pd.read_excel(os.path.join(home, 'SPEI_liao.xls'), 'sheet2')
data1 = data1.to_numpy()
data2 = data2.to_numpy()
x3 = data1[1:, 10] - (min(data1[1:, 10]) - 1) * np.ones((len(data1) - 1,))
x4 = data2[[i for i in range(21, len(data2))][::12], 6]
u3 = UnivariateDistribution(x3, stats.genextreme)
v4 = UnivariateDistribution(x4, stats.norm)
cdata_uv = np.array([u3.data_cdf, v4.data_cdf]).T
c1 = CopulaDistributionBivariate(cdata_uv, copulas.bivariate.Frank())
c1.plot()