# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
# 径流数据处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numba
from numba import jit
import more_itertools as mit
my_font = font_manager.FontProperties(family="SimHei")

# 读取日径流数据
home = "H:/工作/榆林项目20200826-2017年陕西省教育厅重点实验室及基地项目/径流数据/"
daily_r = pd.read_excel('H:/工作/榆林项目20200826-2017年陕西省教育厅重点实验室及基地项目/径流数据/日径流.xlsx', 0,
                        index_col=0)

# 插补日数据
daily_r.info()

#   Column  Non-Null Count  Dtype 存在问题的为白家川（使用其他数据集的月径流，后），温家川、神木、王道恒塔需要插补
# 0   丁家沟     18628 non-null  float64
# 1   白家川     13149 non-null  float64
# 2   韩家峁     18628 non-null  float64
# 3   横山      18628 non-null  float64
# 4   绥德      18628 non-null  float64
# 5   赵石窑     18628 non-null  float64
# 6   温家川     18261 non-null  float64
# 7   神木      12418 non-null  float64
# 8   王道恒塔    18597 non-null  float64
sm = daily_r.loc[:, "神木"].values
wjc = daily_r.loc[:, "温家川"].values
wdht = daily_r.loc[:, "王道恒塔"].values
time = daily_r.index


# 构建插补函数
def chabu(d, c, name, rank, threshold, yes=0):
    """
    插补函数：进行数据缺失插补
    d   为待插补站点，数组
    c   为插补参考站点，可以是上下游站，数组
    dc时间一一对应，d中缺少的均为nan格式
    name 图片存储路径
    yes 是否存储图片，0,1
    rank 阶数
    threshold = |d-c|相差超过阈值则认为是异常值
    return:
        d 插补完成的序列
        index_nand 插补值的序号
    """
    # 剔除缺少的，用剩余都有的来做拟合
    index_nan = []
    index_nand = np.argwhere(np.isnan(d) == True).flatten().tolist()  # 待插补列的无效值
    index_nanc = np.argwhere(np.isnan(c) == True).flatten().tolist()  # 参考列的无效值
    np_abnormal = abs(d - c)
    np_abnormal[index_nand] = 0
    np_abnormal[index_nanc] = 0
    index_abnormal = np.argwhere(np_abnormal > threshold).flatten().tolist()  # 异常值, |d-c|相差超过阈值
    index_nan.extend(index_nand)
    index_nan.extend(index_nanc)
    index_nan.extend(index_abnormal)
    index_nan = list(set(index_nan))  # 总无效值
    d_ = np.array([d[i] for i in range(len(d)) if i not in index_nan])
    c_ = np.array([c[i] for i in range(len(c)) if i not in index_nan])
    z, residuals, *_ = np.polyfit(c_, d_, rank, full=True)
    p = np.poly1d(z)
    r2_ = np.array([x * x for x in (d_ - d_.mean()).tolist()])
    r2 = 1 - residuals[0] / r2_.sum()
    plt.figure()
    plt.plot(c_, d_, "o", label="实测")
    c_fit = np.linspace(c_.min(), c_.max(), num=100)
    d_fit = p(c_fit)
    plt.plot(c_fit, d_fit, "r", label="拟合")
    plt.xlabel("参考站点", fontproperties=my_font)
    plt.ylabel("待插补站点", fontproperties=my_font)
    plt.text(1, 1, p.__str__() + "\nr2=" + str(format(r2, '.2f')), weight="bold", fontsize=12)
    plt.legend(prop=my_font)
    plt.show()
    if yes == 1:
        plt.savefig(name)
    for i in index_nand:
        d[i] = p(c[i])
    return d, index_nand


# 构建绘图函数,绘制径流时序图(插补后的)
def figure_plot(time, y, name, index_nan, yes=0):
    '''
    time    时间
    y   要绘制的时间序列
    yes 是否存储图片，0,1
    name 图片存储路径
    index_nan 插补序号
    '''
    plt.figure()
    index_nan_ = [list(group) for group in mit.consecutive_groups(index_nan)]  # 把连续数字分组,因为有些插补不是连续的这样
    # 绘制线图会出问题， 调用了more_itertools来给连续数据分组
    plt.plot(time, y)
    # plt.plot(time[index_nan], y[index_nan], linestyle="", marker="+", color="r", label="插补值")
    for list_ in index_nan_:
        plt.plot(time[list_], y[list_], color="r")
    plt.xticks()
    plt.xlabel("时间", fontproperties=my_font)
    plt.ylabel("径流量/万m3", fontproperties=my_font)
    plt.legend([name[46:-5], "插补值"], prop=my_font)
    plt.show()
    if yes == 1:
        plt.savefig(name)


# 神木站插补温家川
wjc, index_wjc = chabu(wjc, sm, home + "神木站插补温家川站.tiff", rank=1, threshold=10000, yes=1)
figure_plot(time, wjc, home + "温家川站径流.tiff", index_wjc, yes=1)
# 王道恒塔站插补神木站
sm, index_sm = chabu(sm, wdht, home + "王道恒塔站插补神木站.tiff", rank=1, threshold=10000, yes=1)
figure_plot(time, wjc, home + "神木站径流.tiff", index_sm, yes=1)
# 神木站插补王道恒塔站
wdht, index_wdht = chabu(wdht, sm, home + "神木站插补王道恒塔站.tiff", rank=1, threshold=10000, yes=1)
figure_plot(time, wjc, home + "王道恒塔站径流.tiff", index_wdht, yes=1)

# 插补后数据传入daily_r
daily_r.loc[:, "神木"] = sm
daily_r.loc[:, "温家川"] = wjc
daily_r.loc[:, "王道恒塔"] = wdht

# 重采样为月径流数据和年径流数据
monthly_r = daily_r.resample("M").sum()
yearly_r = daily_r.resample("A").sum()


# 读取白家川月径流并传入monthly_r中
monthly_r_bjc = pd.read_excel('H:/工作/榆林项目20200826-2017年陕西省教育厅重点实验室及基地项目/径流数据/日径流.xlsx', 4,
                              index_col=0, header=0)
monthly_r_bjc = monthly_r_bjc.iloc[:-3, :-1].values.flatten().T


# 查看两个数据集的差异，不超过23%，认为可以使用替代
difference_absolute = monthly_r_bjc[228:] - monthly_r.loc["1975-01-31":, "白家川"].values
difference_relative = (monthly_r_bjc[228:] - monthly_r.loc["1975-01-31":, "白家川"].values)/monthly_r.loc["1975-01-31":, "白家川"]\
    .values
print(difference_absolute.max(), difference_relative.max()*100, '%', "\n", difference_absolute .min(), difference_relative.min()*100, '%')
# 1723.8399999999965 22.481667260126194 %
#  -1284.264000000001 -17.536770489224487 %


# 月径流传入日径流重采样
monthly_r.loc[:, "白家川"] = monthly_r_bjc[48:]


# monthly_r写入excel
monthly_r.to_excel("H:/工作/榆林项目20200826-2017年陕西省教育厅重点实验室及基地项目/径流数据/月径流.xlsx")
