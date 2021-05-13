# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import sys
import you_get
from you_get import common
import os

# url = input("视频地址：")
# directory = input("视频保存路径：")
home = ''
pages = list(range(42, 56))
url = [f'https://www.bilibili.com/video/BV1B7411L7Qt?p={page}' for page in pages]
directory = [os.path.join('F:/学习/bilibili', f'Tensorflow2.0_page_{page}') for page in pages]

for i in range(len(pages)):
    sys.argv = ["you-get", "-o", directory[i], url[i]]
    common.main()