#coding:utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import arcpy
import os
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )

# for root, dirs, files in os.walk(u'F:/小论文2/代码/study'.decode('utf-8'), topdown=False):
#     if u"kivy" in dirs:
#         dirs.remove(u"kivy")
#     for name in files:
#         print(os.path.join(root, name))
#     for name in dirs:
#         print(os.path.join(root, name))

for root, dirs, files in arcpy.da.Walk(u'F:/小论文2/代码/study'.decode('utf-8'), topdown=False):
    # if u"kivy" in dirs:
    #     dirs.remove(u"kivy")
    for name in files:
        print(os.path.join(root, name))
    for name in dirs:
        print(os.path.join(root, name))

