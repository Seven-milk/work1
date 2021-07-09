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

        fig.show()

    return r


def divideLen(Len, section_num):
    ''' divide a Len into section num
    e.g. len(x) = 10 -> divideLen(len(x), 3) -> [0, 3, 6, 11]
    x = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    for i in range(section_num):
        x_section = x[output[i]: output[i+1]]
        print(x_section)
    >>
    [0 1 2]
    [3 4 5]
    [6 7 8 9]

    e.g. len(x) = 10  -> divideLen(len(x), 5) -> [0, 2, 4, 6, 8, 11]
    x = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>
    [0 1]
    [2 3]
    [4 5]
    [6 7]
    [8 9]

    note: do not use x[output[i]] x[output[i+1]]
    '''
    section = np.linspace(0, Len, section_num + 1, dtype=int).tolist()
    section[-1] += 1
    return section


def extractIndexArray(row, col, array):
    ''' extract All data from array based on two list/array index instead of cross data

    cross:
        array[[0,1,2], [3,4,5]] -> array[0, 3], array[1, 4], array[2, 5]

    extractIndexArray:
        extractIndexArray([0,1,2], [3,4,5], array) -> array in

        row 0 : [0, 3], [0, 4], [0, 5]
        row 1: [1, 3], [1, 4], [1, 5]
        row 2: [2, 3], [2, 4], [2, 5]

    input:
        row/col: list/array, index for extract
        array: np.ndarray

    output:
        ret: the array for input array in row x col
    '''
    index = np.meshgrid(row, col)
    ret = array[index[0], index[1]].T
    return ret


def side(series, x):
    ''' find side of x in series

    series:
        0.1 0.3 0.5 0.4 0.8 0.2 0.0

    x:
        0.2

    side:
        0.1 (0.2) 0.3: index=0 index=1
        0.8 (0.2) 0.2: index=4 index=5
        0.2 (0.2) 0.0: index=5 index=6

    if series is sorted, 0 <= side <= one set
    else 0 <= side
    note: 0.2 == 0.2, it could also be found out
    '''
    series = np.array(series)
    series_l = (series - x)[:-1]
    series_r = (x - series)[1:]
    judge = series_l * series_r
    index = np.where(judge >= 0)[0]
    index = np.delete(index, np.where(index == len(series)))
    ret = [{'index_left': i, 'index_right': i + 1, 'left': series[i], 'right': series[i + 1]} for i in index]

    return ret


def mean_list(lst: list):
    ''' calculate the mean of a list
    input
        lst: a list have int or float.. number
    return
        ret: the mean of the list input
    '''
    if len(lst) != 0:
        ret = sum(lst) / len(lst)
    else:
        ret = 0
    return ret


def date2month(date: int) -> int:
    ''' calculate the month from a int date (19481019 -> 10)
    input
        date: int date, like 19481019
    return
        ret: int month, like 10 (1~12)
    '''
    ret = int((date % 10000 - date % 100) / 100)
    return ret


def date2year(date: int) -> int:
    ''' calculate the year from a int date (19481019 -> 1948)
    input
        date: int date, like 19481019
    return
        ret: int year, like 1948
    '''
    ret = int((date - date % 10000) / 10000)
    return ret


if __name__ == '__main__':
    data = np.arange(10)
    slope_ = slope(data)
    data = [1, 3, 2, 0]
    sortedindex, sorteddata = sortWithIndex(data, p=True)
    r = intersection([0, 1, 2, 2], [1, 2, 2, 0], plot_=True)
    print("r=", r)

    x = list(range(8))
    print(x)
    section = divideLen(8, 3)
    print(section)
    for i in range(3):
        x_section = x[section[i]: section[i+1]]
        print(x_section)

    y = np.random.randint(0, 10, (10, 10))
    print(y)
    print(extractIndexArray([0, 1, 2], [3, 4, 5], y))

    series = [0.1, 0.3, 0.5, 0.4, 0.8, 0.2, 0.0]
    x = 0.2
    ret = side(series, x)
    print(ret)