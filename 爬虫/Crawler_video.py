# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import sys
import you_get
from you_get import common

url = input("视频地址：")
directory = input("视频保存路径：")

sys.argv = ["you-get", "-o", directory, url]
common.main()