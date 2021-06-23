# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# reading and handle esa_cci_sm_v05.2 data
import os
from datetime import datetime
from esa_cci_sm.interface import CCI_SM_025Img
from esa_cci_sm.interface import CCI_SM_025Ds
import numpy.testing as nptest
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import map_plot

# general set
home = 'H:/data_zxd/ESA_CCI_SM_v06.1'


def test():
    # read several parameters
    parameter = 'sm'

    # the class is initialized with the exact filename.
    image_path = os.path.join(home, 'test')
    image_file = 'ESACCI-SOILMOISTURE-L3S-SSMS-ACTIVE-19910807000000-fv05.2.nc'
    img = CCI_SM_025Img(os.path.join(image_path, image_file), parameter=parameter)
    # reading returns an image object which contains a data dictionary
    # with one array per parameter. The returned data is a global 0.25 degree
    # image/array.
    image = img.read()

    plt.pcolormesh(image.lon, image.lat, image.data["sm"])

# sm
parameter = 'sm'
img = CCI_SM_025Ds(data_path=os.path.join(home, '_down/2_daily_images/combined'), parameter=parameter)
fig, ax = plt.subplots()
# range(len(date_period))


image = img.read(datetime(2020, 1, 29, 0))
ax.pcolormesh(image.lon, image.lat, image.data["sm"])

