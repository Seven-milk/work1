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
from matplotlib.image import imread

# <editor-fold, desc="读取数据到pd">
data = pd.read_excel("./出图.xlsx", sheet_name=0)
print(data.info())
# </editor-fold>


# <editor-fold, desc="绘图函数">
def create_map():
    shp_path = './cn_shp/Province_9/'
    # --创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(figsize=(6, 8), dpi=400)  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})

    # --设置地图属性
    provinces = cfeat.ShapelyFeature(
        Reader(shp_path + 'Province_9.shp').geometries(),
        proj, edgecolor='k',
        facecolor='none'
    )
    # 加载省界线
    ax.add_feature(provinces, linewidth=0.6, zorder=2)
    # 加载分辨率为50的海岸线
    ax.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    # 加载分辨率为50的河流~
    ax.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)
     # 加载分辨率为50的湖泊
    ax.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)
    ax.set_extent([105, 133, 15, 45])
    # ax.stock_img()
    ax.imshow(
        imread('./NE1_50M_SR_W.tif'),
        origin='upper',
        transform=proj,
        extent=[-180, 180, -90, 90]
    )
    # --设置网格点属性
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1.2,
        color='k',
        alpha=0.5,
        linestyle='--'
    )
    gl.xlabels_top = False  # 关闭顶端的经纬度标签
    gl.ylabels_right = False  # 关闭右侧的经纬度标签
    gl.xformatter = LONGITUDE_FORMATTER  # x轴设为经度的格式
    gl.yformatter = LATITUDE_FORMATTER  # y轴设为纬度的格式
    gl.xlocator = mticker.FixedLocator(np.arange(95, 145 + 5, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(-5, 45 + 5, 5))

    # --设置小地图
    left, bottom, width, height = 0.67, 0.15, 0.23, 0.27
    ax2 = fig.add_axes(
        [left, bottom, width, height],
        projection=proj
    )
    ax2.add_feature(provinces, linewidth=0.6, zorder=2)
    ax2.add_feature(cfeat.COASTLINE.with_scale('50m'), linewidth=0.6, zorder=10)
    ax2.add_feature(cfeat.RIVERS.with_scale('50m'), zorder=10)
    ax2.add_feature(cfeat.LAKES.with_scale('50m'), zorder=10)
    ax2.set_extent([105, 125, 0, 25])
    # ax2.stock_img()
    ax2.imshow(
        imread('./NE1_50M_SR_W.tif'),
        origin='upper',
        transform=proj,
        extent=[-180, 180, -90, 90]
    )
    return ax
# </editor-fold>

def main():
    ax = create_map()
    title = f'distribution of station around China'
    ax.set_title(title, fontsize=18)

    df = pd.read_csv('buyo_position.csv')
    df['lon'] = df['lon'].astype(np.float64)
    df['lat'] = df['lat'].astype(np.float64)
    ax.scatter(
        df['lon'].values,
        df['lat'].values,
        marker='o',
        s=10 ,
        color ="blue"
    )
    for i, j, k in list(zip(df['lon'].values, df['lat'].values, df['name'].values)):
        ax.text(i - 0.8, j + 0.2, k, fontsize=6)
    plt.savefig('station_distribute_map.png')

if __name__ == '__main__':
    main()