# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 批量修改met文件中的NUM_METGRID_SOIL_LEVELS，使其=4
import netCDF4 as nc
from netCDF4 import Dataset
import os

home = "H:/data/NECP_FNL_ds083.2/met/"
dir_met = os.listdir(home)
dir_met = [home + dir_ for dir_ in dir_met]



# for dir_alone in dir_met:
#     rootgrp = Dataset(dir_alone, "a")
#     rootgrp.NUM_METGRID_SOIL_LEVELS = 4
#     rootgrp.close()
#
# # 概览
# rootgrp = Dataset(dir_met[0], "r")
# print(rootgrp.variables.keys())
# print('****************************')
# print(rootgrp)
# print('****************************')
# print(rootgrp.ncattrs())
# print('****************************')
# print(rootgrp.NUM_METGRID_SOIL_LEVELS)
# rootgrp.close()

# rename,win下不能执行，因为路径名不能包含:，需要去linux下执行
# for dir_alone in dir_met:
#     new_dir = dir_alone.replace("^%", ":")
#     os.rename(dir_alone, new_dir)