# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
import pandas as pd
import numpy as np

# 读取日径流数据并重采样为月径流数据
daily_r = pd.read_excel('H:/工作/榆林项目20200826-2017年陕西省教育厅重点实验室及基地项目/径流数据/日径流.xlsx', 0,
                        index_col=0)
monthly_r = daily_r.resample("M").sum()
yearly_r = daily_r.resample("A").sum()


# 读取白家川月径流并传入monthly_r中
monthly_r_bjc = pd.read_excel('H:/工作/榆林项目20200826-2017年陕西省教育厅重点实验室及基地项目/径流数据/日径流.xlsx', 4,
                              index_col=0, header=0)
# monthly_r_bjc = monthly_r_bjc.T
monthly_r_bjc = monthly_r_bjc.iloc[:-3, :-1].values.flatten().T


# 查看两个数据集的差异，不超过23%，认为可以使用替代
difference_absolute = monthly_r_bjc[228:] - monthly_r.loc["1975-01-31":, "白家川"].values
difference_relative = (monthly_r_bjc[228:] - monthly_r.loc["1975-01-31":, "白家川"].values)/monthly_r.loc["1975-01-31":, "白家川"]\
    .values
print(difference_absolute.max(), difference_relative.max()*100, '%', "\n", difference_absolute .min(), difference_relative.min()*100, '%')
# 1723.8399999999965 22.481667260126194 %
#  -1284.264000000001 -17.536770489224487 %

# 替代
monthly_r.loc[:, "白家川"] = monthly_r_bjc[48:]