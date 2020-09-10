# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 常见函数、脚本的总结，以后可以直接import用
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader, natural_earth
import pandas as pd
import matplotlib.ticker as mticker
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numba
from numba import jit
import more_itertools as mit



# <editor-fold, desc="绘地图">
def create_map(title):
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


def plot_data(ax, array_data_lon, array_data_lat, array_data, title, cmap_name='YlOrBr'):
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
