# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import rasterio
import rasterstats
from rasterio.plot import show
from matplotlib import pyplot as plt
import os

home = "F:/data/NDVI"
tiff_path = os.path.join(home, "MODND1F.20000306.CN.NDVI.MAX.V2.TIF")
rf = rasterio.open(tiff_path, mode="r")

print(rf.meta)
show(rf)
plt.show()