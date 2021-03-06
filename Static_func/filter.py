# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# filter
import abc
import scipy.signal as signal
import numpy as np
import draw_plot
import matplotlib.pyplot as plt


class FilterBase(abc.ABC):
    ''' FilterBase abstract class '''

    @abc.abstractmethod
    def filtering(self, data, **kwargs):
        ''' define filtering abstract ethod '''

    @abc.abstractmethod
    def plot(self):
        ''' define plot abstarct method '''


class ButterFilter(FilterBase):
    ''' Butterworth Filter '''

    def __init__(self, data, N=3, Wn=0.5, **kwargs):
        ''' init function
        input:
            data: data series, which will be filtered, it could be 1D array or 2D array(such as shape(2, 200), if you
                  want to use array with shape(200, 2), set axis=0 in self.filtering)
                  1D array: x=np.arange(len(self._data)), y=self._data
                  2D array: x=self._data[0, :], y=self._data[1, :], shape as (2, 200), col: samples, row: variables
            N: int, The order of the filter.
            Wn : array_like, The critical frequency or frequencies.
            **kwargs: key word args, which used in signal.butter init, it could contain "btype", "analog", "output"

            reference *signal.butter*

        output:
            self.butter: (b, a), Numerator (`b`) and denominator (`a`) polynomials of the IIR filter.
            self.filtered_data: filtered data, shape same as data
        '''

        self._data = data
        self.butter = signal.butter(N, Wn, **kwargs)
        self.filtered_data = self.filtering(self._data)

    def filtering(self, data, **kwargs):
        ''' Implement FilterBase.filtering
        input:
            data: data series, which will be filtered
            **kwargs: key word args, which used in signal.filtfilt function, it could contain "axis", "padtype",
                     "padlen", "method", "irlen"

            reference *signal.filtfilt*

        output:
            filtered_data: data series after filtering
        '''

        filtered_data = signal.filtfilt(*self.butter, data, **kwargs)

        return filtered_data

    def plot(self, time_ticks=None, labelx="Time", labely="Data", **kwargs):
        ''' Implement FilterBase.plot
        input:
            time_ticks: dict {"ticks": ticks, "interval": interval}, the ticks of time, namely axis x
                        note: for ticks: len=len(data), for interval: dtype=int
            labelx/y: the labelx/y of the first ax

            **kwargs: keyword args of subplots, keyword args in Figure init function
        '''
        # define time
        time = np.arange(len(self._data))

        fig = draw_plot.Figure(**kwargs)
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="Filter", labelx=labelx, labely=labely, legend_on=True)

        # plot data & filter_data
        if len(self._data.shape) == 1:
            line_data = draw_plot.PlotDraw(time, self._data, color="gray", linewidth=0.6, label="Data")
            line_filtered_data = draw_plot.PlotDraw(time, self.filtered_data, color="r", linewidth=0.6,
                                                    label="Filtered Data")

        else:
            line_data = draw_plot.PlotDraw(self._data[0, :], self._data[1, :], color="gray", linewidth=0.6,
                                           label="Data")
            line_filtered_data = draw_plot.PlotDraw(self.filtered_data[0, :], self.filtered_data[1, :], color="r",
                                                    linewidth=0.6,
                                                    label="Filtered Data")

        draw.adddraw(line_data)
        draw.adddraw(line_filtered_data)

        # set ticks while time_ticks!=None
        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])


class MAFilter(FilterBase):
    ''' Moving average method Filter '''

    def __init__(self, data, window: int = 5, **kwargs):
        ''' init function
        input:
            data: data series, which will be filtered, it could be 1D array or 2D array(such as shape(2, 200), if you
                  want to use array with shape(200, 2), set axis=0 in self.filtering)
            window: filter window, it should be odd > 1, indicating point numbers that window contains to calculate avg
            **kwargs: key word args, which used in np.convolve

            reference *np.convolve*

        output:
            self.filtered_data: filtered data, shape same as data
        '''

        self._data = data
        self.window = window
        if self.window % 2 == 0:
            raise ValueError("window shoule be odd")

        self.filtered_data = self.filtering(self._data, **kwargs)

    def filtering(self, data, **kwargs):
        ''' Implement FilterBase.filtering
        [1, 2, 3, 4, 5] [2, 3_, 4, 5, 6][3_, 4_, 5, 6, 7]...[., ., ., ., .] : window = 5
        window1 -> avg = 3_ -> replace 3
        window2 -> avg = 4_ -> replace 4
        ...

        input:
            data: data series, which will be filtered
            **kwargs: key word args, which used in np.convolve

        output:
            filtered_data: data series after filtering
        '''

        filtered_data = np.copy(data)
        n_ingore = self.window // 2

        if len(data.shape) == 1:
            filtered_data[n_ingore:-n_ingore] = np.convolve(data, np.repeat(1 / self.window, self.window), mode="valid",
                                                            **kwargs)

        else:
            for i in range(data.shape[0]):
                filtered_data[i, n_ingore:-n_ingore] = np.convolve(data[i, :], np.repeat(1 / self.window, self.window),
                                                                   mode="valid", **kwargs)

        return filtered_data

    def plot(self, time_ticks=None, labelx="Time", labely="Data", **kwargs):
        ''' Implement FilterBase.plot
                input:
                    time_ticks: dict {"ticks": ticks, "interval": interval}, the ticks of time, namely axis x
                                note: for ticks: len=len(data), for interval: dtype=int
                    labelx/y: the labelx/y of the first ax

                    **kwargs: keyword args of subplots, keyword args in Figure init function
                '''
        # define time
        time = np.arange(len(self._data))

        fig = draw_plot.Figure(**kwargs)
        draw = draw_plot.Draw(fig.ax, fig, gridy=True, title="Filter", labelx=labelx, labely=labely, legend_on=True)

        # plot data & filter_data
        if len(self._data.shape) == 1:
            line_data = draw_plot.PlotDraw(time, self._data, color="gray", linewidth=0.6, label="Data")
            line_filtered_data = draw_plot.PlotDraw(time, self.filtered_data, color="r", linewidth=0.6,
                                                    label="Filtered Data")

        else:
            line_data = draw_plot.PlotDraw(self._data[0, :], self._data[1, :], color="gray", linewidth=0.6,
                                           label="Data")
            line_filtered_data = draw_plot.PlotDraw(self.filtered_data[0, :], self.filtered_data[1, :], color="r",
                                                    linewidth=0.6,
                                                    label="Filtered Data")

        draw.adddraw(line_data)
        draw.adddraw(line_filtered_data)

        # set ticks while time_ticks!=None
        if time_ticks != None:
            plt.xticks(time[::time_ticks["interval"]], time_ticks["ticks"][::time_ticks["interval"]])


if __name__ == '__main__':
    x = np.hstack((sorted(np.random.rand(100, ) * 10), sorted(np.random.rand(100, ) * 100)))
    y = np.hstack((sorted(np.random.rand(100, ) * 20), sorted(np.random.rand(100, ) * 300)))

    z = np.vstack((x, y))
    # z = x

    # bf ButterFilter
    # bf = ButterFilter(z)
    # bf.plot()

    # maf MAfilter
    # maf = MAFilter(x, window=11)
    # mafd = maf.filtered_data
    # maf.plot()
