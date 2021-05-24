# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import abc


class DistributionBase(abc.ABC):
    ''' distribution Base abstract class '''
    @abc.abstractmethod
    def fit(self, data):
        ''' fit '''

    @abc.abstractmethod
    def cdf(self, data):
        ''' cdf '''

    @abc.abstractmethod
    def pdf(self, data):
        ''' pdf '''
