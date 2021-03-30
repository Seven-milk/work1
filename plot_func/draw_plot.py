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
        self.set(font_label=self.font_label, font_ticks=self.font_ticks)
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

    def set(self, font_label, font_ticks, font_family='Times New Roman'):
        ''' set the fig '''
        config = {'font.family': font_family, 'font.size': font_label["size"]}
        plt.rcParams.update(config)
        plt.xticks(fontproperties=font_ticks)
        plt.yticks(fontproperties=font_ticks)

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
                        labels: set the box labels
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


class Draw:
    ''' Add Draw in one ax, this class is used to represent ax and plot a draw '''

    def __init__(self, ax, Fig: Figure, gridx=False, gridy=False, title="Draw", labelx=None, labely=None, **kwargs):
        ''' init function
        input:
            ax: a single ax for this map from Figure.ax[i]
            fig: Figure, the Figure.fig contain this ax, implement the communication between Map and Fig (for plot colobar)
            gridx/y: bool, whether to open the grid lines
            labelx/y: str, default=None, plot the label of x and y
            title: title of this ax
            **kwargs: keyword args of this ax, it could contain "xlim"[=(0,10)] "ylim" " "xlabel" ...
        '''
        self.ax = ax
        self.Fig = Fig
        self.gridx = gridx
        self.gridy = gridy
        self.labelx = labelx
        self.labely = labely
        self.title = title
        self.kwargs = kwargs
        self.set(gridx=self.gridx, gridy=self.gridy, title=self.title, labelx=self.labelx, labely=self.labely,
                 **self.kwargs)

    def adddraw(self, draw: DrawBase):
        ''' add draw
        input:
            draw: DrawBase class, it can be the sub class of Drawbase: such as BaseMap, RasterMap, ShpMap...
        '''
        draw.plot(self.ax, self.Fig)

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


if __name__ == "__main__":
    # np.random.seed(15)
    f = Figure(3)
    for i in range(3):
        x = np.random.rand(100, 3)
        d = Draw(f.ax[i], f, gridy=True, labelx="X", labely="Y")
        box = BoxDraw(x, labels=['x1', 'x2', 'x3'])
        d.adddraw(box)
