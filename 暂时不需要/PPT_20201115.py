# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# PPT出图，slope
import os
import numpy as np
import pandas as pd
from scipy import stats
import re
import openpyxl

# read file
home = "H:/工作/20201115PPT"
path = [p for p in os.listdir(home) if p[:7]=="drought"]
result = np.full((10, 3), fill_value=0, dtype="float")

for i in range(len(path)):
    read = np.loadtxt(os.path.join(home, path[i]), dtype="int")
    year = read[:, 0]
    duration = read[:, 4]
    formation = read[:, 5]
    recovery = read[:, 6]
    result[i, 0] = stats.linregress(range(len(duration)), duration)[0]
    result[i, 1] = stats.linregress(range(len(formation)), formation)[0]
    result[i, 2] = stats.linregress(range(len(recovery)), recovery)[0]
result_pd = pd.DataFrame(result, index=[re.search(r"subbasin_\d{1,2}", path[i])[0] for i in range(len(path))], columns=["持续时间slope", "形成时间slope", "恢复时间slope"])
result_pd.to_excel(os.path.join(home, "slope.xlsx"))
