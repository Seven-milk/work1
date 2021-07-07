# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 珠江项目，csv格式改变
import pandas as pd
import os
import numpy as np

home = 'H:/work/黄生志/2021珠江项目/数据汇交'
df = pd.read_csv(os.path.join(home, 'Result 13_[And] Time-invariant bivariate risk of a 20-yr design drought.csv'), index_col=0)
index_ = []
column_ = []
value_ = []

for index in df.index:
    for column in df.columns:
        index_.append(index)
        column_.append(column)
        value_.append(df.loc[index, column])

combine = np.vstack([np.array(index_), np.array(column_), np.array((value_))]).T
out = pd.DataFrame(combine, columns=["index", "column", "value"])
# out = out.set_index(["index", "column"])
out.to_csv("out.csv")