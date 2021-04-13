# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
import numpy as np
import os

home = "H:/work/jianglong/jianglong_work2"
stand_path = "stander.xlsx"
data_paths = os.listdir(os.path.join(home, "csv"))
stand = pd.read_excel(os.path.join(home, stand_path))


def fun(start, end, suffix, dir):
    filename = []
    for year in range(start, end + 1):
        for month in range(1, 13):
            if month < 10:
                filename.append(f"{year}_0{month}{suffix}")
            else:
                filename.append(f"{year}_{month}{suffix}")

    filename = [os.path.join(dir, filename_) for filename_ in filename]

    for i in range(data.shape[1]):
        data_ = data[:, i] * 100
        stand["Value"] = data_
        stand["Value"] = stand["Value"].astype(int)
        stand.to_csv(filename[i], index=False)


for data_path in data_paths:
    data = np.loadtxt(os.path.join(home, "csv/" + data_path), dtype=float, delimiter=",")
    data = data.T
    start, end = int(data_path[:4]), int(data_path[5:9])
    suffix = data_path[9:]
    dir = os.path.join(home, data_path[:-4])
    if not os.path.exists(dir):
        os.mkdir(dir)
    fun(start, end, suffix, dir)
