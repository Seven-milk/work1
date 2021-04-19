# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# define basical math function
import numpy as np
import draw_plot

def slope(data):
    ''' cal slope of data, unit = data unit /interval unit
        note: slope[0]=0
    '''
    return data - np.append(data[0], data[:-1])


def sortWithIndex(data, p=False, **kwargs):
    ''' sort data and sort index too '''
    index_ = np.arange(len(data))
    sortedindex = sorted(index_, key=lambda index: data[index], **kwargs)
    sorteddata = sorted(data, **kwargs)

    # print
    if p == True:
        print("Original Data", " " * 5, "Sorted Data")
        print(" index data          index data")
        for i in range(len(data)):
            print(" " * 2, f"{index_[i]}", " " * 3, f"{data[i]}", " " * 10, f"{sortedindex[i]}", " " * 3,
                  f"{sorteddata[i]}")

    return sortedindex, sorteddata

def intersection(line1, line2, plot_=False):
    ''' calculate intersection point from two lines
    x21, y21             x12, y12

               (x, y)

    x11, y11             x22, y22

    (y22 - y21) * x + (x21 - x22) * y = x21 * y22 - x22 * y21
    (y12 - y11) * x + (x11 - x12) * y = x11 * y12 - x12 * y11

    AX = b
     |(y22 - y21)   (x21 - x22)|  x  |  =  | x21 * y22 - x22 * y21 |
    |(y12 - y11)   (x11 - x12)|  y  |  =  | x11 * y12 - x12 * y11 |

    input:
        line1/2: list, [x11, y11, x12, y12], [x21, y21, x22, y22], that two lines have four points will be calculated
                intersection
        plot_: whether to plot

    output:
        r: np.array[x, y], x = r[0], y = r[1]

    '''
    # unpack
    x11, y11, x12, y12 = line1
    x21, y21, x22, y22 = line2

    A = [[y22 - y21, x21 - x22],
        [y12 - y11, x11 - x12]]

    b = [x21 * y22 - x22 * y21, x11 * y12 - x12 * y11]

    r = np.linalg.solve(A, b)

    if plot_ == True:
        fig = draw_plot.Figure()
        draw = draw_plot.Draw(fig.ax, fig, gridx=True, gridy=True, title="Find Intersection Point", labelx="X",
                              labely="Y", legend_on=True)
        line1_plot = draw_plot.PlotDraw([line1[0], line1[2]], [line1[1], line1[3]], color="k", alpha=0.5, label="line1")
        line2_plot = draw_plot.PlotDraw([line2[0], line2[2]], [line2[1], line2[3]], color="k", alpha=0.5, label="line2")

        intersection_plot = draw_plot.PlotDraw(r[0], r[1], "ro", markersize=3, label="intersection point")
        intersection_Text = draw_plot.TextDraw(f"({r[0]}, {r[1]})", [r[0], r[1]], color="r")

        line11_Text = draw_plot.TextDraw(f"({line1[0]}, {line1[1]})", [line1[0], line1[1]], color="r")
        line12_Text = draw_plot.TextDraw(f"({line1[2]}, {line1[3]})", [line1[2], line1[3]], color="r")
        line21_Text = draw_plot.TextDraw(f"({line2[0]}, {line2[1]})", [line2[0], line2[1]], color="r")
        line22_Text = draw_plot.TextDraw(f"({line2[2]}, {line2[3]})", [line2[2], line2[3]], color="r")

        draw.adddraw(line1_plot)
        draw.adddraw(line2_plot)
        draw.adddraw(intersection_plot)
        draw.adddraw(intersection_Text)
        draw.adddraw(line11_Text)
        draw.adddraw(line12_Text)
        draw.adddraw(line21_Text)
        draw.adddraw(line22_Text)

    return r

if __name__ == '__main__':
    data = np.arange(10)
    slope_ = slope(data)
    data = [1, 3, 2, 0]
    sortedindex, sorteddata = sortWithIndex(data, p=True)
    r = intersection([1, 1, 2, 2], [1, 2, 2, 1], plot_=True)
