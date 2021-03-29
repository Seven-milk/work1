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

    def __init__(self, addnumber: int = 1, dpi: int = 200):
        ''' init function
        input:
            addnumber: the init add fig number
            dpi: figure dpi, default=300

        self.figNumber: fig number in the base map, default=1
        self.figRow: the row of subfigure, default=1
        self.figCol: the col of subfigure, default=1

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
        self.fig = plt.figure(dpi=self.dpi)
        self.add = False
        self.addFig(addnumber)
        self.font_label = {'family': 'Times New Roman', 'weight': 'normal',
                           'size': 8 if isinstance(self.ax, np.ndarray) else 10}
        self.font_ticks = {'family': 'Times New Roman', 'weight': 'normal',
                           'size': 8 if isinstance(self.ax, np.ndarray) else 10}
        self.font_title = {'family': 'Times New Roman', 'weight': 'bold',
                           'size': 10 if isinstance(self.ax, np.ndarray) else 15}
        plt.rcParams['font.size'] = self.font_label["size"]
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.xticks(fontproperties=self.font_ticks)
        plt.yticks(fontproperties=self.font_ticks)
        if self.add == True:
            self.unview_last()

    def addFig(self, AddNumber=1):
        ''' add blank figure and return ax '''
        self.figNumber += AddNumber
        if self.figNumber >= 2:
            self.calrowcol()
        self.fig.clf()
        self.ax = self.fig.subplots(nrows=self.figRow, ncols=self.figCol)
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

    def plot(self, ax, Fig):
        ''' '''


class Draw:
    ''' Add Draw in one ax, this class is used to represent ax and plot a draw '''

    def __init__(self, ax, Fig: Figure, gridx=False, gridy=False, title="Draw", **kwargs):
        ''' init function
        input:
            ax: a single ax for this map from Figure.ax[i]
            fig: Figure, the Figure.fig contain this ax, implement the communication between Map and Fig (for plot colobar)
            gridx/y: bool, whether to open the grid lines
            title: title of this ax
            **kwargs: keyword args of this ax, it could contain "xlim"[=(0,10)] "ylim" " "xlabel" ...
        '''
        self.ax = ax
        self.Fig = Fig
        self.gridx = gridx
        self.gridy = gridy
        self.title = title
        self.kwargs = kwargs
        self.set(gridx=self.gridx, gridy=self.gridy, title=self.title, **self.kwargs)

    def adddraw(self, draw: DrawBase):
        ''' add draw
        input:
            draw: DrawBase class, it can be the sub class of Drawbase: such as BaseMap, RasterMap, ShpMap...
        '''
        draw.plot(self.ax, self.Fig)

    def set(self, gridx=False, gridy=False, title="Draw", **kwargs):
        ''' set this Draw(ax)
        input:
            grid: bool, whether to open the grid lines
            title: title of this ax
        '''
        # set
        self.ax.set(**kwargs)
        # grid
        self.ax.grid(gridx, axis="x")
        self.ax.grid(gridy, axis="y")
        # title
        self.ax.set_title(title, fontdict=self.Fig.font_title)