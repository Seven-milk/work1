# code: utf-8
# author: "XudongZheng"
# email: Z786909151@163.com
import netCDF4
import numpy
import pandas
from netCDF4 import Dataset
from datetime import datetime, timedelta
from netCDF4 import num2date, date2num

rootgrp = Dataset("F:/GLDAS_NOAH025_M.A194801.020.nc4", "a")
print(rootgrp)
print("***********")
print(rootgrp.groups)
print("***********")
fcstgrp = rootgrp.createGroup("forecasts")
analgrp = rootgrp.createGroup("analyses")
print(rootgrp.groups)
print("***********")


def walktree(top):
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walktree(value):
            yield children


for children in walktree(rootgrp):
    for child in children:
        print(children)

print("************")
print(rootgrp.dimensions)
print("************")
print(rootgrp.variables)
print("************")

# level = rootgrp.createDimension("level", None)
# levels = rootgrp.createVariable("level", "i4", ("level",))
time_variable = rootgrp.variables["time"]
lat_variable = rootgrp.variables["lat"]
temp = rootgrp.variables["temp"]
# temp = rootgrp.createVariable("temp", "f4", ("time", "level", "lat", "lon",))
# levels[:] = [1000., 850., 700., 500., 300., 250., 200., 150., 100., 50.]
#a = numpy.array([[[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]], [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]])

dates = [datetime(2001,3,1)+n*timedelta(hours=12) for n in range(temp.shape[0])]
# times[:] = date2num(dates, units=times.units, calendar=times.calendar)

rootgrp.close()