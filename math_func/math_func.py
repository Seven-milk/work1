# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# define basical math function
import numpy as np


def slope(data):
    ''' cal slope of data, unit = data unit /interval unit
        note: slope[0]=0
    '''
    return data - np.append(data[0], data[:-1])


def sortWithIndex(data, p=False, **kwargs):
    ''' sort data and sort index too '''
    index_ = np.arange(len(data))
    sortedindex = sorted(index_, key=lambda index: data[index], **kwargs)
    sorteddata = sorted(data, **kwargs)

    # print
    if p == True:
        print("Original Data", " " * 5, "Sorted Data")
        print(" index data          index data")
        for i in range(len(data)):
            print(" " * 2, f"{index_[i]}", " " * 3, f"{data[i]}", " " * 10, f"{sortedindex[i]}", " " * 3,
                  f"{sorteddata[i]}")

    return sortedindex, sorteddata


if __name__ == '__main__':
    data = np.arange(10)
    slope_ = slope(data)
    data = [1, 3, 2, 0]
    sortedindex, sorteddata = sortWithIndex(data, p=True)
