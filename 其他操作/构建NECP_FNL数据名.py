# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 由于原始程序不能Import pd所以需要额外写一个来计算名称
import pandas as pd

# grib1 19990730_18_00-20071206_06_00
date1 = pd.date_range('19990730', '20071207', freq='6H').strftime("%Y%m%d_%H").tolist()[3:-3]
filelist_grib1=[f'grib1/{date[:4]}/{date[:4]}.{date[4:6]}/fnl_{date}_00.grib1' for date in date1]
# grib2 20071206_12_00-20200921_18_00
date2 = pd.date_range('20071206', '20200922', freq='6H').strftime("%Y%m%d_%H").tolist()[2:-1]
filelist_grib2=[f'grib1/{date[:4]}/{date[:4]}.{date[4:6]}/fnl_{date}_00.grib1' for date in date2]
# 合并
filelist_grib1.extend(filelist_grib2)
filelist = filelist_grib1

with open('../necp数据路径.txt', 'w') as f:
    f.write(" ".join(filelist))