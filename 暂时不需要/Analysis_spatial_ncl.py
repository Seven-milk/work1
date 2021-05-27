# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
# plot the spatial distribution of FD
import numpy as np
import pandas as pd
import FlashDrought
import os
import re
from matplotlib import pyplot as plt
import Ngl

# general set
home = "/home/z786909151/FD_data"
data_path = os.path.join(home, "SoilMoist_RZ_tavg.txt")
coord_path = os.path.join(home, "coord.txt")
coord = pd.read_csv(coord_path, sep=",")
sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
date = pd.date_range('19480101', '20141230', freq='d').strftime("%Y%m%d").to_numpy(dtype="int")

# time avg data
sm_rz_time_avg = sm_rz.mean(axis=0)

# plot the spatial distribution of time avg data
lon = coord.loc[:, "lon"].values
lat = coord.loc[:, "lat"].values
wks = Ngl.open_wks("png", "plot_spatial_distribution_of_avg_data")
res = Ngl.Resources()
res.nglFrame = False  # Don't advance frame.
res.cnFillOn = True
res.cnFillMode = "RasterFill"
res.cnLinesOn = False

res.sfXArray = lon
res.sfYArray = lat
res.mpLimitMode = "LatLon"
res.tiMainString = "plot_spatial_distribution_of_avg_data"
res.tiMainFont = "Helvetica-Bold"
res.tiXAxisString = "lon"
res.tiYAxisString = "lat"
res.mpMinLatF = int(min(lat))
res.mpMinLonF = int(min(lon))
res.mpMaxLatF = int(max(lat)) + 1
res.mpMaxLonF = int(max(lon)) + 1
# res.cnLineLabelsOn = False
# res.lbOrientation = "horizontal"
# res.mpGridLatSpacingF = 1
# res.mpGridLonSpacingF = 1
plot = Ngl.contour_map(wks, sm_rz_time_avg, res)
Ngl.frame(wks)
Ngl.end()


