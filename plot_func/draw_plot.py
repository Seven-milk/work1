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

    def __init__(self, addnumber: int = 1, dpi: int = 300, figsize=(12, 5), wspace=None, hspace=None, family="Arial",
                 figRow=1, figCol=1, axflatten=True, **kwargs):
        ''' init function
        input:
            addnumber: the init add fig number
            dpi: figure dpi, default=300
            figsize: figure size, default=(12, 5)
            wspace/hspace: the space between subfig
            family: font family
            figRow=1, figCol=1: to set the fig row and col
            axflatten: whether flatten the ax
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
        self.figRow = figRow
        self.figCol = figCol
        self.dpi = dpi
        self.kwargs = kwargs
        self.figsize = figsize
        self.fig = plt.figure(dpi=self.dpi, figsize=self.figsize)
        self.add = False
        self.axflatten = axflatten
        self.wspace = wspace
        self.hspace = hspace
        self.addFig(addnumber, wspace=self.wspace, hspace=self.hspace, **self.kwargs)
        self.font_label = {'family': family, 'weight': 'normal',
                           'size': 6 if isinstance(self.ax, np.ndarray) else 8}
        self.font_ticks = {'family': family, 'weight': 'normal',
                           'size': 6 if isinstance(self.ax, np.ndarray) else 8}
        self.font_title = {'family': family, 'weight': 'bold',
                           'size': 6 if isinstance(self.ax, np.ndarray) else 8}
        self.font_legend = {'family': family, 'weight': 'bold',
                            'size': 4 if isinstance(self.ax, np.ndarray) else 5}
        if self.add == True:
            self.unview_last()

    def addFig(self, AddNumber=1, wspace=None, hspace=None, **kwargs):
        ''' add blank figure and return ax '''
        self.figNumber += AddNumber
        if self.figNumber >= 2:
            self.calrowcol()
        self.fig.clf()
        self.ax = self.fig.subplots(nrows=self.figRow, ncols=self.figCol, **kwargs)
        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
        if isinstance(self.ax, np.ndarray) and self.axflatten:
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
        if not os.path.exists(os.path.join(os.getcwd(), 'fig')):
            os.mkdir(os.path.join(os.getcwd(), 'fig'))
        plt.savefig('fig/' + title + '.jpg', dpi=self.dpi, bbox_inches='tight')

    def show(self):
        ''' show fig '''
        plt.show()


class FigureVert(Figure):
    ''' figure set in vertical direction '''

    def addFig(self, AddNumber=1, wspace=None, hspace=None, **kwargs):
        ''' add blank figure and return ax, override Figure.addFig() '''
        self.figNumber += AddNumber
        self.fig.clf()
        self.ax = self.fig.subplots(nrows=self.figNumber, ncols=1, **kwargs)
        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
        if isinstance(self.ax, np.ndarray):
            self.ax = self.ax.flatten()


class FigureHorizon(Figure):
    ''' figure set in Horizon direction, override Figure.addFig() '''

    def addFig(self, AddNumber=1, wspace=None, hspace=None, **kwargs):
        ''' add blank figure and return ax '''
        self.figNumber += AddNumber
        self.fig.clf()
        self.ax = self.fig.subplots(nrows=1, ncols=self.figNumber, **kwargs)
        self.fig.subplots_adjust(wspace=wspace, hspace=hspace)
        if isinstance(self.ax, np.ndarray):
            self.ax = self.ax.flatten()


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

    def __init__(self, *args, cb_label=None, **kwargs):
        ''' init function
        input:
            *args: position args, it should contain x, y: values in vector to draw scatter plot
            cb_label: the colorbar label, when cmap is set, draw the colorbar, it define specified separaly because it
                      not belong **kwargs in ax.scatter
            **kwargs: keyword args, it could contain
                       "marker";
                       "c"[=colors], "s"[=sizes] [c&s can plot 3D scatter, color & size represent z axis, when colors
                                     is set as z axis, should use the "cmap" params and plot the color bar];
                       "color", "s" [color & s can only set by on value], "alpha";
                       "cmap"(such as "viridis"), "label" [legend label];
                       "label": legend label;

        '''
        self.args = args
        self.cb_label = cb_label
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        pc = ax.scatter(*self.args, **self.kwargs)

        # plot colorbar: "cmap" should in self.kwargs.keys()
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


class PlotDraw(DrawBase):
    ''' Plot Draw (2D) '''

    def __init__(self, *args, **kwargs):
        ''' init function
        input:
            *args: position args, it should contain [x]y: values in vector to draw plot plot, ...
                   it could contain "fmt" - '[marker][line][color]'
            **kwargs: keyword args, it could contain "alpha", "color", "visible", "linestyle" "label"(legend label)

        '''
        self.args = args
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        ax.plot(*self.args, **self.kwargs)


class HistDraw(DrawBase):
    ''' Hist Draw (2D) '''

    def __init__(self, x, label=None, **kwargs):
        ''' init function
        input:
            x: values in vector to draw hist plot
            label: legend label
            **kwargs: keyword args, it could contain "bins", "alpha", "cumulative"
                bins suggest: when use kde, set bins = int(len(x) * kde.covariance_factor())
        '''
        self.x = x
        self.label = label
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implement the DrawBase.plot function '''
        ax.hist(self.x, label=self.label, **self.kwargs)


class BarDraw(DrawBase):
    ''' Bar Draw '''

    def __init__(self, x, height, **kwargs):
        ''' init function
        input:
            x: The x coordinates of the bars. See also align for the alignment of the bars to the coordinates.
            height: The height(s) of the bars.
            **kwargs: keyword args, it could contain "width", "bottom", "aligh", "color", "edgecolor", "linewidth",
                "tick_label", "label", see pyplot.bar()
        '''
        self.x = x
        self.height = height
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        ''' Implement the DrawBase.plot function '''
        ax.bar(self.x, self.height, **self.kwargs)


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
                  Draw(ax_twinx, Fig) to draw kde, follow the example

            -------- example -------------------------------------------------------------------------------------
            d4 = Draw(f.ax[4], f, gridy=True, labelx="X", labely="number", legend_on={"loc": "upper right",
                    "framealpha": 0.8}, title="Hist & Kde")
            d4.adddraw(h)
            d5 = Draw(f.ax[4].twinx(), f, gridy=False, labelx=None, labely="PDF", legend_on={"loc": "upper left",
                    "framealpha": 0.8}, title=None)
            d5.adddraw(k)
            ------------------------------------------------------------------------------------------------------
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


class PcolormeshDraw(DrawBase):
    ''' Pcolormesh Draw '''

    def __init__(self, *args, cb_label="cb", **kwargs):
        ''' init function
        cb_label: colorbar label name
        *args: position args
        **kwargs: keyword args, it could contain "alpha", "norm", "cmap", "vmin", "vmax", "shading", "antialiased",
                reference ax.pcolormesh
        general use: pcolormesh([X, Y,] C, **kwargs)
        note: attention to the relationship between C and the axis X/Y, sometimes, to harmonize them, you should use
                np.flip(C, axis=..)
        '''
        self.args = args
        self.kwargs = kwargs
        self.cb_label = cb_label

    def plot(self, ax, Fig):
        ''' Implements the DrawBase.plot function '''
        # pcolormesh
        pcm = ax.pcolormesh(*self.args, **self.kwargs)

        # colorbar
        # shrinkrate = 0.9 if isinstance(Fig.ax, np.ndarray) else 1
        # extend = 'neither' if isinstance(Fig.ax, np.ndarray) else 'both'
        cb = Fig.fig.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05)  # , shrink=shrinkrate, extend=extend
        cb.ax.tick_params(labelsize=Fig.font_label["size"])  # , direction='in'
        if isinstance(Fig.ax, np.ndarray):
            cb.ax.set_title(label=self.cb_label, fontdict=Fig.font_label)
        else:
            cb.set_label(self.cb_label, fontdict=Fig.font_label)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_family('Arial')


class ContourDraw(DrawBase):
    ''' Contour Draw '''

    def __init__(self, *args, **kwargs):
        ''' init function
        *args: position args, it could contain a int to control lines number(it can also be specificed in kwargs,
                "levels"), i.e. contour([X, Y,] C, 20, **kwargs)
        **kwargs: keyword args, it could contain "colors", "cmap", "linewidths", "alpha", "vmin", "vmax", "norm",
                "extent", "extend", reference ax.contour
        general use: contour([X, Y,] C, **kwargs), note: X, Y = np.meshgrid(x, y) should be preprocessed before plot
        note: if you put contour above a contourf, set the same level can make it looks better
        '''
        self.args = args
        self.kwargs = kwargs

    def plot(self, ax, Fig):
        # contour
        c = ax.contour(*self.args, **self.kwargs)

        # clabel
        ax.clabel(c, inline=True, fontsize=Fig.font_label["size"]-1)  # 2.5


class ContourfDraw(DrawBase):
    ''' Contourf Draw '''

    def __init__(self, *args, cb_label="cb", **kwargs):
        ''' init function
         cb_label: colorbar label name
        *args: position args
        **kwargs: keyword args, it could contain "colors", "cmap", "linewidths", "alpha", "vmin", "vmax", "norm",
                "extent", "extend", reference ax.contourf
        '''
        self.args = args
        self.kwargs = kwargs
        self.cb_label = cb_label

    def plot(self, ax, Fig):
        # contourf
        cf = ax.contourf(*self.args, **self.kwargs)

        # colorbar
        # shrinkrate = 0.9 if isinstance(Fig.ax, np.ndarray) else 1
        # extend = 'neither' if isinstance(Fig.ax, np.ndarray) else 'both'
        cb = Fig.fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.05)  # , shrink=shrinkrate, extend=extend
        cb.ax.tick_params(labelsize=Fig.font_label["size"])  # , direction='in'
        if isinstance(Fig.ax, np.ndarray):
            cb.ax.set_title(label=self.cb_label, fontdict=Fig.font_label)
        else:
            cb.set_label(self.cb_label, fontdict=Fig.font_label)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_family('Arial')


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
                      "upper right", "framealpha": 0.8}, when legend_on=True(it set as default)
            **kwargs: keyword args of this ax, it could contain "xlim"[=(0,10)] "ylim" ...
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
            self.set_legend(**{"loc": "upper right", "framealpha": 0.8})
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
    f = Figure(figRow=2, figCol=5, hspace=0.5, wspace=0.5)
    facecolors = ["lightgrey", 'lightgreen', 'lightblue']  # pink
    x = np.random.rand(500, 3)
    # d0: box and text
    d0 = Draw(f.ax[0], f, gridy=True, labelx="X", labely="Y", title="BoxDraw, TextDraw")
    box = BoxDraw(x, labels=['x1', 'x2', 'x3'], patch_artist=True, facecolors=facecolors)
    text = TextDraw("text", extent=[3, 0.9], color="r")
    d0.adddraw(box)
    d0.adddraw(text)
    # d1: scatter
    d1 = Draw(f.ax[1], f, gridy=True, labelx="X", labely="Y", legend_on={"loc": "upper right", "framealpha": 0.8},
              title="ScatterDraw")
    s = ScatterDraw(x[:, 0], x[:, 1], cb_label="cb", label="x0-x1-x2", marker="+", c=x[:, 2], s=x[:, 2] * 10,
                    cmap="viridis", alpha=0.5)
    d1.adddraw(s)
    # d2: hist
    d2 = Draw(f.ax[2], f, gridy=True, labelx="X", labely="number", legend_on=True, title="HistDraw")  # legend_on=True,
    # set as default
    h = HistDraw(x[:, 0], label="hist", bins=100, alpha=0.5)
    d2.adddraw(h)
    # d3: kde
    d3 = Draw(f.ax[3], f, gridy=True, labelx="X", labely="PDF", legend_on=True, title="KdeDraw")
    k = KdeDraw(x[:, 0], inter_num=500, label="kde", color="b", sample_on=True)
    d3.adddraw(k)
    # d4/5: hist and kde
    d4 = Draw(f.ax[4], f, gridy=True, labelx="X", labely="number", legend_on=True, title="Hist & Kde")
    d4.adddraw(h)
    d5 = Draw(f.ax[4].twinx(), f, gridy=False, labelx=None, labely="PDF", legend_on=True, title=None)
    d5.adddraw(k)
    # d6: line
    d6 = Draw(f.ax[5], f, gridy=True, labelx="X", labely="Y", legend_on=True, title="PlotDraw")
    l = PlotDraw(sorted(x[:, 0]), sorted(x[:, 1]), "b--", label="x0-x1")  # color="b", linestyle="--"
    d6.adddraw(l)
    # d7: bar
    d7 = Draw(f.ax[6], f, gridy=True, labelx="X", labely="Y", legend_on=True, title="BarDraw")
    bar = BarDraw([1, 2, 3], x[0, :3], label="x")
    d7.adddraw(bar)
    # d8: pcolormesh
    d8 = Draw(f.ax[7], f, gridy=True, labelx="X", labely="Y", legend_on=False, title="Pcolormesh")
    pcolormesh = PcolormeshDraw(x[:100, 0].reshape((10, 10)))
    d8.adddraw(pcolormesh)
    # d9: contour
    d9 = Draw(f.ax[8], f, gridy=True, labelx="X", labely="Y", legend_on=False, title="Contour")
    contour = ContourDraw(x[:100, 0].reshape((10, 10)), colors="k", linewidths=1)
    d9.adddraw(contour)
    # d10: contourf
    d10 = Draw(f.ax[9], f, gridy=True, labelx="X", labely="Y", legend_on=False, title="Contourf")
    contourf = ContourfDraw(x[:100, 0].reshape((10, 10)))
    d10.adddraw(contourf)

    # del d8
    # f.unview_last()
    # fig show
    f.show()