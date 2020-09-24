#coding:utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 郭怿：shp裁剪tiff
import arcpy
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

home = u"H:/data/基础数据资料/原始tif/"
# store = u"H:/data/基础数据资料/郭怿NDVI/新建文件地理数据库.gdb/"
arcpy.env.workspace = home
# featureclasses = arcpy.ListFeatureClasses()
# rasters = arcpy.ListRasters()
#
# for raster in rasters:
#     arcpy.Clip_management(in_raster=raster, rectangle="94 31 112 42", out_raster="clip_" + raster[-10:]
#                           , in_template_dataset=featureclasses[0]
#                           , nodata_value=-9999
#                           , clipping_geometry=True
#                           , maintain_clipping_extent=True)
rasters = arcpy.ListRasters(u"clip*")
mean_raster=[]
for raster in rasters:
    r = arcpy.Raster(raster)
    mean_raster.append(r.mean)
print(mean_raster)
with open(home+"mean.txt", 'w') as f:
    f.write(" ".join([str(x) for x in mean_raster]))
