# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# define basical math function
import numpy as np


def slope(data):
    ''' cal slope of data, unit = data unit /interval unit '''
    return data - np.append(data[0], data[:-1])
