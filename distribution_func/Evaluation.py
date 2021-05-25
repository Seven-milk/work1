# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# distribution test and evaluation
from scipy import stats
import numpy as np
import math
import Univariatefit
import Nonparamfit


class Evaluation:

    def kstest(self, data, theory_cdf, **kwargs):
        ''' kstest
        input:
            data : array_like, the data to evaluation
            theory_cdf : str, array_like or callable
                If array_like, it should be a 1-D array of observations of random
                variables, and the two-sample test is performed (and rvs must be array_like)
                If a callable, that callable is used to calculate the cdf. (recommend, such as UnivariateDistribution.cdf)
                If a string, it should be the name of a distribution in `scipy.stats`,
                which will be used as the cdf function.
            **kwargs: key words args, it could contain see more from stats.kstest

        output:
            ks_ret = (KS-statistic, p_value): p_value > 1-alpha, passed
        '''
        ks_ret = stats.kstest(data, theory_cdf, **kwargs)
        return ks_ret

    def aic(self, data, theory_ppf, param_num):
        # rmse
        rmse = self.rmse(data, theory_ppf)
        mse = rmse ** 2

        # aic
        n = len(data)
        aic = 2 * param_num + n * math.log(mse)

        return aic

    def rmse(self, data, theory_ppf):
        # Empirical Distribution CDF
        ed = Nonparamfit.EmpiricalDistribution()
        ed_cdf = ed.cdf(data)

        # inverse of cdf
        theory_data = theory_ppf(ed_cdf)

        # rmse
        n = len(data)
        rmse = (sum((data - theory_data) ** 2) / n) ** 0.5

        return rmse

    def ppplot(self):
        pass

    def qqplot(self):
        pass


if __name__ == '__main__':
    # np.random.seed(10)
    data = np.random.normal(0, 1, 1000)
    norm_ = Univariatefit.UnivariateDistribution(stats.norm)
    norm_.fit(data)
    norm_.plot(data)
    evaluation = Evaluation()
    kstest_ret_norm = evaluation.kstest(data, norm_.cdf)

    print(f"kstest\nKS-statistic={kstest_ret_norm[0]}, p_value={kstest_ret_norm[1]}")

    rmse = evaluation.rmse(data, norm_.ppf)
    aic = evaluation.aic(data, norm_.ppf, len(norm_.params))

    print(f'rmse: {rmse}, aic: {aic}')