# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# check whether the file is downloaded completely
#　思路：还原ｕｒｌ或将其转为ｎａｍｅ＋集合差集
import os
import re
import requests

home = "G:/GLDAS_NOAH"
URL = os.path.join(home, "subset_GLDAS_NOAH025_3H_2.0_20210328_114227.txt")