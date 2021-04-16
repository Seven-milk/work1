# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import numpy as np
import scipy
import abc
import draw_plot


# Define CurveBase class
class CurveBase(abc.ABC):
    ''' CurveBase abstract class '''

    @abc.abstractmethod
    def fit(self):
        ''' fit Curve '''


class PolyCurve(CurveBase):
    ''' Poly Curve fit based on np '''

    def __init__(self, x, y, show=True, *args, **kwargs):
        ''' init function
        input:
            x: x value for fit, 1D array (M, )
            y: y value for fit curve, 1D/2D array (M, ) or (M, K)[fit multiple curves which share x]
            show: bool, whether excute show()
            *args: position args
            **kwargs: keyword args, it could contain:
                    "deg": degree of the fitting polynomial
                    "w": weight
        output:
            self.pcf: np.ndarray, shape (deg + 1,) or (deg + 1, K), Polynomial coefficients, highest power first.
               If y was 2-D, the coefficients for k-th data set are in p[:,k]
            self.residuals: residuals for this fit
            self.pnm : A one-dimensional polynomial class


        '''
        self.x = x
        self.y = y
        self.args = args
        self.kwargs = kwargs
        self.fit()
        if show == True:
            self.show()

    def fit(self):
        ''' Implement CurveBase.fit function '''
        self.pcf, self.residuals, *_ = np.polyfit(self.x, self.y, full=True, *self.args, **self.kwargs)
        self.pnm = np.poly1d(self.pcf)

    def show(self):
        ''' show fit information '''
        print("Fitting Curve:\n", self.pnm)
        print("Residuals: ", self.residuals, "\n")

    def getValue(self, x):
        ''' get the curve value in x
        input:
            x: it could be a numpy array, a value, or a one-dimensional polynomial class
        '''
        return np.polyval(self.pnm, x)

    def getLinespace(self, num=1000):
        ''' '''
        x_linespace = np.linspace(min(self.x), max(self.x), num)
        y_linespace = self.getValue(x_linespace)
        return x_linespace, y_linespace

    def plot(self, num=1000):
        ''' plot fit curve
        input:
            num: linespace point number
        '''
        fig = draw_plot.Figure()
        draw = draw_plot.Draw(fig.ax, fig, gridx=True, gridy=True, title="Curve Fitting", labelx="X", labely="Y",
                              legend_on=True)
        plot_original = draw_plot.PlotDraw(self.x, self.y, "b.", alpha=0.3, label="Original data")
        x_linespace, y_linespace = self.getLinespace(num=num)
        plot_fitcurve = draw_plot.PlotDraw(x_linespace, y_linespace, "r", label=f"Least square fit")
        draw.adddraw(plot_original)
        draw.adddraw(plot_fitcurve)
        return fig, draw


if __name__ == '__main__':
    x = np.arange(100)
    y = np.random.rand(100, )
    polycurve_deg1 = PolyCurve(x, y, deg=1)
    polycurve_deg5 = PolyCurve(x, y, deg=5)
    polycurve_deg1.plot()
    polycurve_deg5.plot()

    polycurve_deg1.getValue(10)
