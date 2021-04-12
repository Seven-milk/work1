# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
import numpy as np
import os

home = "F:/work/jianglong/data"
stand_path = "stander.xlsx"
data_path = "2001-2005_max.csv"

stand = pd.read_excel(os.path.join(home, stand_path))
data = np.loadtxt(os.path.join(home, data_path), dtype=float)
filename = [f"{year}_{month}_max.csv" for year in range(2001, 2006) for month in range(1, 13)]
filename = [os.path.join(home, filename_) for filename_ in filename]

for i in range(data.shape[1]):
    data_ = data[:, i] * 100
    stand["Value"] = data_
    stand["Value"] = stand["Value"].astype(int)
    stand.to_csv(filename[i], index=False)

