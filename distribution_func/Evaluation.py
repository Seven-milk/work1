# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# distribution test and evaluation
from scipy import stats
import numpy as np
import math
import Univariatefit


class Evaluation:

    def kstest(self, data, theory_dis, **kwargs):
        ''' kstest
        input:
            data : array_like, the data to evaluation
            theory_dis : str, array_like or callable
                If array_like, it should be a 1-D array of observations of random
                variables, and the two-sample test is performed (and rvs must be array_like)
                If a callable, that callable is used to calculate the cdf. (recommend, such as UnivariateDistribution.cdf)
                If a string, it should be the name of a distribution in `scipy.stats`,
                which will be used as the cdf function.
            **kwargs: key words args, it could contain see more from stats.kstest

        output:
            ks_ret = (KS-statistic, p_value): p_value > 1-alpha, passed
        '''
        ks_ret = stats.kstest(data, theory_dis, **kwargs)
        return ks_ret

    def ppplot(self):
        pass

    def qqplot(self):
        pass


if __name__ == '__main__':
    # np.random.seed(10)
    data = np.random.normal(0, 1, 1000)
    norm = Univariatefit.UnivariateDistribution(stats.norm)
    norm.fit(data)
    norm.plot(data)
    evaluation = Evaluation()
    kstest_ret_norm = evaluation.kstest(data, norm.cdf)

    print(f"kstest\nKS-statistic={kstest_ret_norm[0]}, p_value={kstest_ret_norm[1]}")