# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import abc
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib
from scipy import stats


# Define MapBase class
class DrawBase(abc.ABC):
    ''' DrawBase abstract class '''

    @abc.abstractmethod
    def plot(self, ax, Fig):
        ''' plot Draw '''


class Figure:
    ''' figure set '''

    def __init__(self, addnumber: int = 1, dpi: int = 200, wspace=0.2, hspace=0.4, **kwargs):
        ''' init function
        input:
            addnumber: the init add fig number
            dpi: figure dpi, default=300
            wspace/hspace: the space between subfig
            **kwargs: keyword args of subplots, it could contain "sharex" "sharey"

        self.figNumber: fig number in the base map, default=1
        self.figRow: the row of subfigure, default=1
        self.figCol: the col of subfigure, default=1
        self.kwargs: keyword args of subplots

        Main output: self.ax, a list of subfig in a canvas used to plot
        note:
        1) if the fig close, call Figure.fig.show() function
        2) if figNumber is odd by addfig, it will be set to even by self.calrowcol()
        3) if figNumber==1, dont use Figure.ax[0], just use Figure.ax
        '''
        self.figNumber = 0
        self.figRow = 1
        self.figCol = 1
        self.dpi = dpi
        self.kwargs = kwargs
        self.fig = plt.figure(dpi=self.dpi)
        self.add = False
        self.wspace = wspace
        self.hspace = hspace
        self.addFig(addnumber, wspace=self.wspace, hspace=self.hspace, **self.kwargs)
        self.font_label = {'family': 'Times New Roman', 'weight': 'normal',
                           'size': 12 if isinstance(self.ax, np.ndarray) else 15}
        self.font_ticks = {'family': 'Times New Roman', 'weight': 'normal',
                           'size': 12 if isinstance(self.ax, np.ndarray) else 15}
        self.font_title = {'family': 'Times New Roman', 'weight': 'bold',
                           'size': 15 if isinstance(self.ax, np.ndarray) else 18}
        self.font_legend = {'family': 'Times New Roman', 'weight': 'bold',
                            'size': 8 if isinstance(self.ax, np.ndarray) else 10}
        if self.add == True:
            self.unview_last()

    def addFig(self, AddNumber=1, wspace=0.2, hspace=0.4, **kwargs):
        ''' add blank figure and return ax '''
        self.figNumber += AddNumber
        if self.figNumber >= 2:
            self.calrowcol()
        self.fig.clf()
        self.ax = self.fig.subplots(nrows=self.figRow, ncols=self.figCol, **kwargs)
        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
        if isinstance(self.ax, np.ndarray):
            self.ax = self.ax.flatten()

    def calrowcol(self, rowfirst=True):
        ''' Decomposition factor of self.figNumber to get self.figRow and self.figCol
            rowfirst: row first calculating
        '''
        # if self.figNumber == 2
        if self.figNumber == 2:
            self.figRow = 2
            self.figCol = 1
            if rowfirst == False:
                self.figRow, self.figCol = self.figCol, self.figRow
            return

        # Determine if self.figNumber is prime and decomposition it
        while True:
            # prime
            for i in range(2, self.figNumber):
                if not self.figNumber % i:  # remainder equal to 0
                    self.figRow = i
                    self.figCol = self.figNumber // self.figRow
                    if rowfirst == False:
                        self.figRow, self.figCol = self.figCol, self.figRow
                    return
            # not prime: self.figNumber + 1 (blank subplot)
            self.figNumber += 1
            self.add = True

    def unview_last(self):
        ''' unview the last ax '''
        self.ax[-1].set_visible(False)

    def reset(self):
        ''' reset Figure to the init state, the canvas is still exist that can be used '''
        self.fig.clf()
        self.figNumber = 0
        self.figRow = 1
        self.figCol = 1
        self.addFig()

    def resetax(self, num=0, colorbar_num=0):
        ''' reset ax to the init state, num start from 0, the ax is still exist that can be used '''
        ax_ = [ax for ax in self.fig.get_axes() if ax._label != "<colorbar>"]
        ax_[num].cla()
        ax_[num].outline_patch.set_visible(False)
        ax_bar = [ax for ax in self.fig.get_axes() if ax._label == "<colorbar>"]
        if len(ax_bar) != 0:
            ax_bar[colorbar_num].remove()

    def save(self, title):
        ''' save fig
        input:
            title: the title to save figure
        '''
        plt.savefig('./fig/' + title + '.jpg', dpi=self.dpi, bbox_inches='tight')


class BoxDraw(DrawBase):
    ''' box plot draw '''

    def __init__(self, x, violin=False, facecolors=None, **kwargs):
        ''' init function
        input:
            x: Array or a sequence of vectors, such as: [box_1, box_2, box_3, box_4]
            violin=False: whether to plot violin draw
            facecolors: list, ["lightgrey", 'lightgreen', 'lightblue'], set the facecolor of box, it requires
                        patch_artist==True
            **kwargs: keyword args, it could contain ["labels"]!!, "vert", "notch" "zorder" "meanline", "showmeans",
                    "showbox", reference ax.boxplot
                        labels: set the box labels in x axis
                        vert: set the box draw direction
                        notch: set the box notched
                        showfliers: whether show the fliers
        '''
        self.x = x
        self.violin = violin
        self.facecolors = facecolors
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        if self.violin == True:
            bx = ax.violinplot(self.x, **self.kwargs)
        else:
            bx = ax.boxplot(self.x, **self.kwargs)
        # fill with colors
        if self.facecolors != None and (self.kwargs["patch_artist"] == True):
            for patch, color in zip(bx['boxes'], self.facecolors):
                patch.set_facecolor(color)


class TextDraw(DrawBase):
    ''' Text Draw '''

    def __init__(self, text: str, extent: list, **kwargs):
        ''' init function
        input:
            text: str, the text to plot
            extent: list of two elements, [x, y], define the position to plot text
            **kwargs: keyword args, it could contain "color" "fontdict"(dict) "alpha" "zorder" ...
        '''
        self.text = text
        self.extent = extent
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        # define the default fontdict
        if "fontdict" not in self.kwargs.keys():
            self.kwargs["fontdict"] = Fig.font_label
        ax.text(self.extent[0], self.extent[1], self.text, **self.kwargs)


class ScatterDraw(DrawBase):
    ''' Scatter Draw (2D) '''

    def __init__(self, x, y, label=None, cb_label=None, **kwargs):
        ''' init function
        input:
            x/y: values in vector to draw scatter plot
            label: legend label
            cb_label: the colorbar label, when cmap is set, draw the colorbar
            **kwargs: keyword args, it could contain "marker", "c=colors", "s=sizes" [c&s can plot 3D scatter,
                      color & size represent z axis, when colors is set as z axis, should use the "cmap" params and
                      plot the color bar], "color", "s" [color & s can only set by on value], "alpha",
                       "cmap"(such as "viridis")

        '''
        self.x = x
        self.y = y
        self.label = label
        self.cb_label = cb_label
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        pc = ax.scatter(self.x, self.y, label=self.label, **self.kwargs)
        if "cmap" in self.kwargs.keys():
            shrinkrate = 1  # 0.7 if isinstance(Fig.ax, np.ndarray) else 0.9
            extend = 'neither' if isinstance(Fig.ax, np.ndarray) else 'both'
            cb = Fig.fig.colorbar(pc, ax=ax, orientation='vertical', shrink=shrinkrate, pad=0.05, extend=extend)
            cb.ax.tick_params(labelsize=Fig.font_label["size"], direction='in')
            if isinstance(Fig.ax, np.ndarray):
                cb.ax.set_title(label=self.cb_label, fontdict=Fig.font_label)
            else:
                cb.set_label(self.cb_label, fontdict=Fig.font_label)
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_family('Times New Roman')


class HistDraw(DrawBase):
    ''' Hist Draw (2D) '''

    def __init__(self, x, label=None, **kwargs):
        ''' init function
        input:
            x: values in vector to draw hist plot
            label: legend label
            **kwargs: keyword args, it could contain "bins", "alpha"
                bins suggest: when use kde, set bins = int(len(x) * kde.covariance_factor())
        '''
        self.x = x
        self.label = label
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        ax.hist(self.x, label=self.label, **self.kwargs)


class KdeDraw(DrawBase):
    ''' Hist Draw (2D) '''

    def __init__(self, x, bw_method="scott", inter_num=100, sample_on=True, label=None, **kwargs):
        ''' init function
        input:
            x: values in vector to draw Kde plot
            bw_method: bandwidth selection method, bw_method='Scott', "silverman"
            inter_num: interpolation number, much number, smoother line
            sample_on: whether to plot samples
            label: legend label
            **kwargs: keyword args, it could contain "bins", "alpha", reference plot

            note: if want to plot kde on ax_twin, ax_twinx= ax.twinx(), create new Draw, and set ax_twinx into
                  Draw(ax_twinx, Fig) to draw kde
        '''
        self.x = x
        self.bw_method = bw_method
        self.sample_on = sample_on
        self.inter_num = inter_num
        self.label = label
        self.kwargs = kwargs
        self.kde = stats.gaussian_kde(x, bw_method=self.bw_method)
        self.x_eval = np.linspace(min(x), max(x), num=int((max(x) - min(x)) * inter_num))

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        # plot samples
        if self.sample_on == True:
            # ax_twinx = ax.twinx()
            ax.scatter(self.x, np.zeros(self.x.shape), marker="|", color="r", linewidths=0.3, s=100, label="Samples")
        # kde plot
        ax.plot(self.x_eval, self.kde(self.x_eval), label=self.label, **self.kwargs)


class Draw:
    ''' Add Draw in one ax, this class is used to represent ax and plot a draw '''

    def __init__(self, ax, Fig: Figure, gridx=False, gridy=False, title="Draw", labelx=None, labely=None,
                 legend_on=False, **kwargs):
        ''' init function
        input:
            ax: a single ax for this map from Figure.ax[i]
            fig: Figure, the Figure.fig contain this ax, implement the communication between Map and Fig (for plot colobar)
            gridx/y: bool, whether to open the grid lines
            labelx/y: str, default=None, plot the label of x and y
            title: title of this ax
            legend_on: bool, whether open the legend, it could also be a dict to set legend, such as legend_on={"loc":
                      "upper right", "framealpha": 0.8}
            **kwargs: keyword args of this ax, it could contain "xlim"[=(0,10)] "ylim" " "xlabel" ...
        '''
        self.ax = ax
        self.Fig = Fig
        self.gridx = gridx
        self.gridy = gridy
        self.labelx = labelx
        self.labely = labely
        self.title = title
        self.legend_on = legend_on
        self.kwargs = kwargs
        self.set(gridx=self.gridx, gridy=self.gridy, title=self.title, labelx=self.labelx, labely=self.labely,
                 **self.kwargs)

    def adddraw(self, draw: DrawBase):
        ''' add draw
        input:
            draw: DrawBase class, it can be the sub class of Drawbase: such as BaseMap, RasterMap, ShpMap...
        '''
        draw.plot(self.ax, self.Fig)
        # legend must after plot
        if self.legend_on == True:
            self.set_legend()
        elif self.legend_on == False:
            return
        else:
            # it can be a dict to set legend
            self.set_legend(**self.legend_on)

    def set(self, gridx=False, gridy=False, title="Draw", labelx=None, labely=None, **kwargs):
        ''' set this Draw(ax)
        input:
            grid: bool, whether to open the grid lines
            title: title of this ax
            labelx: x label
            labely: ylabel
            **kwargs: keyword args of this ax, it could contain "xlim"[=(0,10)] "ylim" " "xlabel" ...
        '''
        # set
        self.ax.set(**kwargs)
        # grid
        if gridx:
            self.ax.grid(gridx, axis="x", alpha=0.5, linestyle='--')
        else:
            self.ax.grid(gridx)
        if gridy:
            self.ax.grid(gridy, axis="y", alpha=0.5, linestyle='--')
        else:
            self.ax.grid(gridy)
        # x/y labels
        self.ax.set_xlabel(labelx, fontdict=self.Fig.font_label, loc="right", labelpad=0.001)
        self.ax.set_ylabel(labely, fontdict=self.Fig.font_label)
        # title
        self.ax.set_title(title, fontdict=self.Fig.font_title)
        # ticks
        self.ax.tick_params(labelsize=self.Fig.font_label["size"], direction='in')
        labels = self.ax.get_xticklabels() + self.ax.get_yticklabels()
        [label.set_font(self.Fig.font_ticks) for label in labels]

    def set_legend(self, **kwargs):
        ''' set the legend
        input:
            **kwargs: keyword args of this legend, it could contain
                "loc"[conbination of upper lower left right]
                "frameon": bool, whether open the fram
                "framalpha": 0~1, frame alpha
                "ncol": int, the col number of the legend
                "shadow": bool, whether open the shadow
                "borderpad", "labelspacing"...
        '''
        self.ax.legend(prop=self.Fig.font_legend, **kwargs)


if __name__ == "__main__":
    # np.random.seed(15)
    f = Figure(6, wspace=0.5)
    facecolors = ["lightgrey", 'lightgreen', 'lightblue']  # pink
    x = np.random.rand(500, 3)
    d0 = Draw(f.ax[0], f, gridy=True, labelx="X", labely="Y", title="BoxDraw, TextDraw")
    box = BoxDraw(x, labels=['x1', 'x2', 'x3'], patch_artist=True, facecolors=facecolors)
    text = TextDraw("text", extent=[3, 0.9], color="r")
    d0.adddraw(box)
    d0.adddraw(text)
    d1 = Draw(f.ax[1], f, gridy=True, labelx="X", labely="Y", legend_on={"loc": "upper right", "framealpha": 0.8},
              title="ScatterDraw")
    s = ScatterDraw(x[:, 0], x[:, 1], label="x0-x1-x2", marker="+", c=x[:, 2], s=x[:, 2] * 10, cmap="viridis",
                    alpha=0.5, cb_label="cb")
    d1.adddraw(s)
    d2 = Draw(f.ax[2], f, gridy=True, labelx="X", labely="number", legend_on={"loc": "upper right", "framealpha": 0.8},
              title="HistDraw")
    h = HistDraw(x[:, 0], label="hist", bins=100, alpha=0.5)
    d2.adddraw(h)
    d3 = Draw(f.ax[3], f, gridy=True, labelx="X", labely="PDF", legend_on={"loc": "upper right", "framealpha": 0.8},
              title="KdeDraw")
    k = KdeDraw(x[:, 0], inter_num=500, label="kde", color="b", sample_on=True)
    d3.adddraw(k)
    d4 = Draw(f.ax[4], f, gridy=True, labelx="X", labely="number", legend_on={"loc": "upper right", "framealpha": 0.8},
              title="Hist & Kde")
    d4.adddraw(h)
    d5 = Draw(f.ax[4].twinx(), f, gridy=False, labelx=None, labely="PDF", legend_on={"loc": "upper left",
                                                                                    "framealpha": 0.8}, title=None)
    d5.adddraw(k)
