# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
import os

root_name = 'F:/文件/zt/20200816返校文件/返校名单'
file_name = os.listdir(root_name)
result = pd.read_excel(root_name + '/' + file_name[0], index_col=0)

for fn in file_name[1:]:
    file_n = root_name + '/' + fn
    data = pd.read_excel(file_n, index_col=0, skiprows=[0, 1, 2])
    result = result.append(data)

result.to_excel(root_name + '/' + "综合.xlsx", index=False)
