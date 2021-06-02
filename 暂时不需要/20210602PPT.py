# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import draw_plot
import matplotlib.pyplot as plt


home = 'H:/work/黄生志/20210601PPT'
DroughtIndex = pd.read_excel(os.path.join(home, 'DroughtIndex.xlsx'))  # 1952~2012
ZS = pd.read_excel(os.path.join(home, '渭河流域灾损数据.xlsx'))
ZS = ZS.iloc[3:64, :]

index_ = np.arange(744, step=12)
DroughtIndex_array = np.zeros((61, 4))
for i in range(len(index_) - 1):
    DroughtIndex_array[i, :] = DroughtIndex.iloc[index_[i]: index_[i + 1], :].mean()

# DroughtIndex_array[:, 0] = DroughtIndex_array[:, 0] * -1
title = ["受灾", "成灾", "绝收"]

fig = draw_plot.Figure(4, family='Microsoft YaHei', wspace=0.4, hspace=0.4)
Year = np.arange(1952, 2013)

for i in range(3):
    draw = draw_plot.Draw(fig.ax[i], fig, gridx=True, gridy=True, title=title[i], labelx="年份", labely="灾损", legend_on=True)
    draw_ = draw_plot.Draw(fig.ax[i].twinx(), fig, gridx=True, gridy=True, title=title[i], labelx="年份", labely="指数值", legend_on=True)
    draw_t = draw_plot.Draw(fig.ax[3], fig, title="corr", xlim=[-1, 1])
    fig.ax[3].axis("off")
    draw.adddraw(draw_plot.BarDraw(Year, ZS.iloc[:, i], label=title[i], color="gray", alpha=0.5))
    for j in range(4):
        draw_.adddraw(draw_plot.PlotDraw(Year, DroughtIndex_array[:, j], label=DroughtIndex.columns[j]))
        corr, p = pearsonr(ZS.iloc[:, i], DroughtIndex_array[:, j])
        draw_t.adddraw(draw_plot.TextDraw(DroughtIndex.columns[j] + "_" + title[i] + "corr=%.2f" % corr + "\np=%.2f" % p, extent=[-1+2/3*i, 0.8-1/4*j], fontdict={"fontsize": 4, "family": "Microsoft YaHei"}))


fig.show()

