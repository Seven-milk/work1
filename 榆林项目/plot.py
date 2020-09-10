# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 处理生成的指数等，使用arcpy进行绘图
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader, natural_earth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# <editor-fold, desc="读取数据到pd">
data = pd.read_excel("./出图.xlsx", sheet_name=0)
print(data.info())


# </editor-fold>


# <editor-fold, desc="绘图函数">
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


# </editor-fold>


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

def main():
    data['lon'] = data['lon'].astype(np.float64)
    data['lat'] = data['lat'].astype(np.float64)
    extend = [107.25, 111.25, 36.75, 39.75]
    det = 0.25
    lat_index, lon_index = index_cal(extend=extend, det=det, data_lat=data['lat'],
                                     data_lon=data['lon'])  # 生成有数据点对于的经纬index
    # 生成完整绘图数组和对应的经纬坐标
    for i in data.iloc[:, 2:].columns:
        array_data, array_data_lon, array_data_lat = array_cal(extend=extend, det=det, lat_index=lat_index,
                                                               lon_index=lon_index, data=data[i])
        ax = create_map(title=i)
        plot_data(ax, array_data_lon, array_data_lat, array_data, title=i, cmap_name='RdBu')


if __name__ == '__main__':
    main()
