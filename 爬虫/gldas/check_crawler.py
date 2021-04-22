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

# open url file and read url in urls
with open(URL, 'r') as file:
    urls = file.read()
    urls = urls.split("\n")

# file_name_all
urls = urls[1:]
file_name_all = [re.search(r"LABEL.*\d\.nc4", url)[0][6:] for url in urls]
file_name_all = [os.path.join(home, file) for file in file_name_all]

# file_name_downloaded
file_name_downloaded = [path for path in os.listdir(home) if path[-4:] == ".nc4"]

# change list to set
file_name_all = set(file_name_all)
file_name_downloaded = set(file_name_downloaded)

# difference set
file_name_notdownloaded = file_name_all - file_name_downloaded

# refactor url
