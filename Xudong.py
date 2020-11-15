# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 常见函数、脚本的总结，以后可以直接import用
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader, natural_earth
import scipy.stats as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager

from PIL import Image

import numba
from numba import jit

import more_itertools as mit

# from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

import os

import math


# <editor-fold, desc="绘地图">
def create_map(title):
    '''基于cartopy绘图
    title 传入图名
    返回
    一个ax句柄，指示绘制的图，用plt.show()就可以看到
    '''
    # --创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(dpi=400)  # 创建页面 figsize=(6, 8),
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

    # --设置地图属性
    # # 加载省界线
    # ax.add_feature(cfeat.ShapelyFeature(Reader('H:/data/中国/国界省界/bou2_4p.shp').geometries(), proj, edgecolor='k',
    #                                     facecolor='none'), linewidth=0.6, zorder=2)
    # # 加载分辨率为50的海岸线
    # ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    # # 加载分辨率为50的河流
    # ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)
    # # 加载分辨率为50的湖泊
    # ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)
    # 加载本地河流和榆林shp
    # ax.add_feature(cfeat.ShapelyFeature(Reader('H:/data/榆林市/river.shp').geometries(), proj,
    #                                       edgecolor='C0', facecolor='none'), linewidth=0.3, zorder=2)
    ax.add_feature(cfeat.ShapelyFeature(Reader('H:/data/榆林市/榆林边界.shp').geometries(), proj,
                                        edgecolor='k', facecolor='none'), linewidth=0.6, zorder=2)

    # 设置范围
    ax.set_extent([107.25, 111.25, 36.75, 39.75])

    # --设置网格点属性
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(106, 112, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(36, 40, 1))
    ax.set_title(title, fontsize=10)
    return ax


def index_cal(extend, det, data_lat, data_lon):
    '''基于extend计算对应的栅格索引
    extend 矩阵，对应lat纬度,lon经度
    det 分辨率，度
    data_lat 数据纬度
    data_lon 数据经度
    data_lat_index 数据纬度对应索引,小到大
    data_lon_index 数据经度对应索引,小到大
    '''
    data_lat_index = [int((data_lat[i] - det / 2 - extend[2]) / det) for i in range(len(data_lat))]
    data_lon_index = [int((data_lon[i] - det / 2 - extend[0]) / det) for i in range(len(data_lon))]

    return data_lat_index, data_lon_index


def array_cal(extend, det, lat_index, lon_index, data):
    '''构建完整数组来绘图
    extend 矩阵，对应lat纬度,lon经度
    det 分辨率，度
    lat_index, lon_index, data 对应有数据点的，lat索引，lon索引，数据值，这需要一一对应
    array_data array_data_lon array_data_lat:补全数据和与之对应的lat和lon，用于绘图
    '''
    array_data = np.full((int((extend[1] - extend[0]) / det + 1), int((extend[3] - extend[2]) / det + 1)),
                         fill_value=-9999,
                         dtype='float')
    mask = array_data == -9999
    array_data = np.ma.array(array_data, mask=mask)
    for i in range(len(lat_index)):
        array_data[lon_index[i], lat_index[i]] = data[i]
    array_data_lon = np.linspace(extend[0], extend[1], num=int((extend[1] - extend[0]) / det + 1)) + det / 2
    array_data_lat = np.linspace(extend[2], extend[3], num=int((extend[3] - extend[2]) / det + 1)) + det / 2
    # +det/2是因为绘图对应的是正方形中心点，而不是正方形边缘
    return array_data, array_data_lon, array_data_lat


def pcolor_data(ax, array_data_lon, array_data_lat, array_data, title, cmap_name='YlOrBr'):
    '''封装绘图函数，传入数据，绘制网格点图
    ax 基于create_map创建的map句柄
    array_data_lon, array_data_lat, array_data 基于array_cal生成的完整绘图数组及与之对应的经纬度数组
    title 用于储存图片的name
    cmap_name 可以用来更换cmap的name，默认为YlOrBr
    '''
    cMap = plt.get_cmap(cmap_name)  # 备选coolwarm
    pc = ax.pcolormesh(array_data_lon, array_data_lat, array_data.T, cmap=cMap)
    cb = plt.colorbar(pc)
    cb.ax.tick_params(labelsize=9)
    plt.rcParams['font.size'] = 9
    # plt.show()
    plt.savefig('./fig/' + title + '.jpg', dpi=350, bbox_inches='tight')
    plt.close()


# </editor-fold>


# <editor-fold, desc="插补数据">
my_font = font_manager.FontProperties(family="SimHei")


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


# </editor-fold>


# <editor-fold, desc="基于basemap绘制地图">
def map_plot(filename, variable_name):
    '''基于basemap绘制地图
    filename 传入nc数据的filename
    variable_name nc中用于绘图的变量名
    返回
        地图
    '''
    map = Basemap()  # 创建basemap对象
    data = Dataset(filename)
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]
    lons, lats = map.makegrid(len(lat), len(lon))
    # lats = lats[::-1]
    data.set_auto_mask(True)
    data1 = data.variables[variable_name][:].reshape(len(lon), len(lat))

    map.drawmapboundary(fill_color='aqua')  # 绘制底图
    # map.fillcontinents(color='#cc9955', lake_color='aqua')  # 绘制陆地
    map.drawcoastlines(linewidth=0.2, color='0.15')  # 绘制海岸线
    map.drawcountries(linewidth=0.2, color='0.15')  # 绘制城市
    map.contourf(lons, lats, data1)  # 绘制数据
    map.colorbar()
    plt.show()


# </editor-fold>


# <editor-fold, desc="MK检验">
def MK_test(x):
    '''MK检验
    x 传入序列, np.ndarray or list
    返回
        (slope, zc1)：分别为slope倾斜度度量和Z统计量
        (统计量反映了序列的差异程度，统计量越大，越可能存在趋势，正负表示正负趋势)
    '''
    s = 0
    length = len(x)
    for m in range(0, length - 1):
        # print(m)
        # print('/')
        for n in range(m + 1, length):
            # print(n)
            # print('*')
            if x[n] > x[m]:
                s = s + 1
            elif x[n] == x[m]:
                s = s + 0
            else:
                s = s - 1
    # 计算vars
    vars = length * (length - 1) * (2 * length + 5) / 18
    # 计算zc
    if s > 0:
        zc = (s - 1) / math.sqrt(vars)
    elif s == 0:
        zc = 0
    else:
        zc = (s + 1) / math.sqrt(vars)

    # 计算za
    zc1 = abs(zc)

    # 计算倾斜度
    ndash = length * (length - 1) // 2
    slope1 = np.zeros(ndash)
    m = 0
    for k in range(0, length - 1):
        for j in range(k + 1, length):
            slope1[m] = (x[j] - x[k]) / (j - k)
            m = m + 1

    slope = np.median(slope1)
    return (slope, zc1)


# </editor-fold>


# <editor-fold, desc="MK检验-标准"> https://github.com/manaruchi/MannKendall_Sen_Rainfall
def mann_kendall(vals, confidence=0.95):
    n = len(vals)

    box = np.ones((len(vals), len(vals)))
    box = box * 5
    sumval = 0
    for r in range(len(vals)):
        for c in range(len(vals)):
            if (r > c):
                if (vals[r] > vals[c]):
                    box[r, c] = 1
                    sumval = sumval + 1
                elif (vals[r] < vals[c]):
                    box[r, c] = -1
                    sumval = sumval - 1
                else:
                    box[r, c] = 0

    freq = 0
    # Lets caluclate frequency now
    tp = np.unique(vals, return_counts=True)
    for tpx in range(len(tp[0])):
        if (tp[1][tpx] > 1):
            tp1 = tp[1][tpx]
            sev = tp1 * (tp1 - 1) * (2 * tp1 + 5)
            freq = freq + sev

    se = ((n * (n - 1) * (2 * n + 5) - freq) / 18.0) ** 0.5

    # Lets calc the z value
    if (sumval > 0):
        z = (sumval - 1) / se
    else:
        z = (sumval + 1) / se

    # lets see the p value

    p = 2 * st.norm.cdf(-abs(z))

    # trend type
    if (p < (1 - confidence) and z < 0):
        tr_type = -1
    elif (p < (1 - confidence) and z > 0):
        tr_type = +1
    else:
        tr_type = 0

    return z, p, tr_type


# </editor-fold>


# <editor-fold, desc="sen-slope"> https://github.com/manaruchi/MannKendall_Sen_Rainfall
def sen_slope(vals, confidence=0.95):
    alpha = 1 - confidence
    n = len(vals)

    box = np.ones((len(vals), len(vals)))
    box = box * 5
    boxlist = []

    for r in range(len(vals)):
        for c in range(len(vals)):
            if (r > c):
                box[r, c] = (vals[r] - vals[c]) / (r - c)
                boxlist.append((vals[r] - vals[c]) / (r - c))
    freq = 0
    # Lets caluclate frequency now
    tp = np.unique(vals, return_counts=True)
    for tpx in range(len(tp[0])):
        if (tp[1][tpx] > 1):
            tp1 = tp[1][tpx]
            sev = tp1 * (tp1 - 1) * (2 * tp1 + 5)
            freq = freq + sev

    se = ((n * (n - 1) * (2 * n + 5) - freq) / 18.0) ** 0.5

    no_of_vals = len(boxlist)

    # lets find K value

    k = st.norm.ppf(1 - (0.05 / 2)) * se

    slope = np.median(boxlist)
    return slope, k, se


# </editor-fold>


# <editor-fold, desc = "基于经纬度读取给定空间范围的nc文件中的变量">
def read_nc(home, coord_file, variable_name, change, start, end, freq, desctype, y=True):
    '''基于经纬度，读取给定空间范围的nc文件中的变量
    home 数据的根目录
    coord_fild 经纬度txt文件的目录
    variable_name 要读取的变量名
    y 是否查看文件基本信息，默认为查看
    start="19480101", end="20141231", freq='M' 数据的开始结束时间和频率
    change 单位转换系数
    desctype 存储格式 txt or xlsx

    返回：
        输出txt or xlsx
    '''
    coord = pd.read_csv(coord_file, sep=",")  # 读取coord榆林的经纬坐标（基于渔网提取）
    coord = coord.round(3)  # 处理单位以便与nc中lat lon一致
    result = os.listdir(home)
    variable_array = np.zeros((len(result), len(coord)))  # 用于储存变量数据的数组

    # <editor-fold desc="获取经纬度对应的索引">
    f1 = Dataset(home + "/" + result[0], 'r')
    Dataset.set_auto_mask(f1, False)
    lat_index_lp = []
    lon_index_lp = []
    lat = f1.variables["lat"][:]
    lon = f1.variables["lon"][:]
    for j in range(len(coord)):
        lat_index_lp.append(np.where(lat == coord["lat"][j])[0][0])
        lon_index_lp.append(np.where(lon == coord["lon"][j])[0][0])
    f1.close()
    # </editor-fold>

    # <editor-fold desc="循环读取对应lp范围的rain变量和runoff变量(基于索引)">
    for i in range(len(result)):
        f = Dataset(home + "/" + result[i], 'r')
        Dataset.set_auto_mask(f, False)
        for j in range(len(coord)):
            variable_array[i, j] = f.variables[variable_name][0, lat_index_lp[j], lon_index_lp[j]]
            # 读取给定经纬度部分
        print(f"第{i}个栅格读取完毕------")
        f.close()
    # </editor-fold>

    # <editor-fold desc="单位处理">
    # 单位转换  "kg m-2 s-1"="mm/s"=3600*24*30"mm/month"，换算系数3600*24*30（按每个月30天算）
    # 单位转换 "kg m-2"="mm",换算系数1
    variable_array = variable_array * change
    # </editor-fold>

    # <editor-fold desc="存储数据">
    if desctype == 'txt':
        np.savetxt(f'{variable_name}_{start}-{end}_{freq}.txt', delimiter=' ')
        np.savetxt('lat_index.txt', lat_index_lp, delimiter=' ')  # 存储经纬索引
        np.savetxt('lon_index.txt', lon_index_lp, delimiter=' ')
        coord.to_csv("coord.txt")
    else:
        # 搭建pd来输出excel
        time = pd.date_range(start, end, freq=freq)
        variable_pd = pd.DataFrame(variable_array, index=time)
        variable_pd = variable_pd.loc[start:end, :]

        # 输出excel
        variable_pd.to_excel(f"{variable_name}_{start}-{end}_{freq}.xlsx")
        coord.to_excel("coord.xlsx")
    # </editor-fold>

    # <editor-fold desc="概览">
    if y == True:
        rootgrp = Dataset(home + "/GLDAS_NOAH025_M.A194801.020.nc4", "r")
        print('keys:', rootgrp.variables.keys())
        print('****************************')
        print('数据：', rootgrp)
        print('****************************')
        print('变量lat：', rootgrp.variables['lat'][:])
        print('****************************')
        print('变量lon：', rootgrp.variables['lon'][:])
        rootgrp.close()
    # </editor-fold>

# </editor-fold>
