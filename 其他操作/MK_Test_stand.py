# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# https://github.com/manaruchi/MannKendall_Sen_Rainfall
import numpy as np
import scipy.stats as st


def mann_kendall(vals, confidence=0.95):
    n = len(vals)

    box = np.ones((len(vals), len(vals)))
    box = box * 5
    sumval = 0
    for r in range(len(vals)):
        for c in range(len(vals)):
            if (r > c):
                if (vals[r] > vals[c]):
                    box[r, c] = 1
                    sumval = sumval + 1
                elif (vals[r] < vals[c]):
                    box[r, c] = -1
                    sumval = sumval - 1
                else:
                    box[r, c] = 0

    freq = 0
    # Lets caluclate frequency now
    tp = np.unique(vals, return_counts=True)
    for tpx in range(len(tp[0])):
        if (tp[1][tpx] > 1):
            tp1 = tp[1][tpx]
            sev = tp1 * (tp1 - 1) * (2 * tp1 + 5)
            freq = freq + sev

    se = ((n * (n - 1) * (2 * n + 5) - freq) / 18.0) ** 0.5

    # Lets calc the z value
    if (sumval > 0):
        z = (sumval - 1) / se
    else:
        z = (sumval + 1) / se

    # lets see the p value

    p = 2 * st.norm.cdf(-abs(z))

    # trend type
    if (p < (1 - confidence) and z < 0):
        tr_type = -1
    elif (p < (1 - confidence) and z > 0):
        tr_type = +1
    else:
        tr_type = 0

    return z, p, tr_type


def sen_slope(vals, confidence=0.95):
    alpha = 1 - confidence
    n = len(vals)

    box = np.ones((len(vals), len(vals)))
    box = box * 5
    boxlist = []

    for r in range(len(vals)):
        for c in range(len(vals)):
            if (r > c):
                box[r, c] = (vals[r] - vals[c]) / (r - c)
                boxlist.append((vals[r] - vals[c]) / (r - c))
    freq = 0
    # Lets caluclate frequency now
    tp = np.unique(vals, return_counts=True)
    for tpx in range(len(tp[0])):
        if (tp[1][tpx] > 1):
            tp1 = tp[1][tpx]
            sev = tp1 * (tp1 - 1) * (2 * tp1 + 5)
            freq = freq + sev

    se = ((n * (n - 1) * (2 * n + 5) - freq) / 18.0) ** 0.5

    no_of_vals = len(boxlist)

    # lets find K value

    k = st.norm.ppf(1 - (0.05 / 2)) * se

    slope = np.median(boxlist)
    return slope, k, se
