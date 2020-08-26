# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 注意：DS表示时间段（累积时段），不是时间点
# import math
import numpy, pandas
from matplotlib import pyplot as plt
import datetime


def run_theory(x: numpy.ndarray, h: float) -> numpy.ndarray:
    '''
    run theory

    Parameter:
        x(numpy.ndarray): time series
        h(float): threshold

    Returns:
        D(numpy.ndarray): Duration
        S(numpy.ndarray): Severity
        flag_start(numpy.ndarray): start number for every event
        flag_end(numpy.ndarray): end number for every event
    '''
    # ------------------------------------------------------------简化，第一种方法耗时太多
    # flag_start = []
    # flag_end = []
    # # 判断第一个点是否为start，当第一个为start时判断第一个是否为end（基于第二个）
    # if x[0] <= h:
    #     flag_start.append(0)
    #     if x[1] > h:
    #         flag_end.append(0)
    # # 中间点判断,要保证内部都小于h
    # for i in range(1, len(x) - 1):
    #     if (x[i] <= h) & (x[i - 1] > h):
    #         flag_start.append(i)
    #     if (x[i] <= h) & (x[i + 1] > h):
    #         flag_end.append(i)
    # # 判断最后一个点是否为end， 当为end时判断是否为开始(基于倒数第二个)
    # if x[len(x) - 1] <= h:
    #     flag_end.append(len(x) - 1)
    #     if x[len(x) - 2] > h:
    #         flag_start.append(len(x) - 1)
    # # 转换为数组
    # flag_start = numpy.array(flag_start)
    # flag_end = numpy.array(flag_end)
    # ------------------------------------------------------------------------
    dry_flag = numpy.argwhere(x <= h).flatten()
    flag_start = dry_flag[numpy.argwhere(dry_flag - numpy.roll(dry_flag, 1).flatten() != 1)].flatten()[1:]
    flag_end = dry_flag[numpy.argwhere(dry_flag - numpy.roll(dry_flag, -1).flatten() != -1)].flatten()[:-1]

    if x[dry_flag[0]] <= h:
        flag_start = numpy.insert(flag_start, 0, dry_flag[0])
        # if x[1] > h:
        #     flag_end = numpy.insert(flag_end, 0, dry_flag[0])
        #  这个方法不需要对end进行第一个值的判断，因为flag_start = dry_flag[numpy.argwhere(dry_flag - numpy.roll(dry_flag, 1)
        #  .flatten() != 1)].flatten()[1:]方法只是flag_start第一个值没判断，flag_end是最后一个值没判断（第一个值判断过了）

    if x[dry_flag[-1]] <= h:
        flag_end = numpy.append(flag_end, dry_flag[-1])
        # if x[dry_flag[-2]] > h:
        #     flag_start = numpy.append(flag_start, dry_flag[-1])

    # 计算历时烈度
    D = flag_end - flag_start + 1
    x = abs(x - h)  # 烈度 = x - h 的绝对值
    # ------------------------------速度慢，用列表推导替代
    # for i in range(len(S)):
    #     S[i] = x[flag_start[i]:flag_end[i] + 1].sum()
    # ---------------------------------
    S = numpy.array([x[flag_start[i]:flag_end[i] + 1].sum() for i in range(len(flag_start))])
    return D, S, flag_start, flag_end


# 已验证


def fun_EP(P, DS=365):
    '''计算有效降水'''
    if type(DS) is numpy.ndarray:  # 当传入DS为数组时
        EP = numpy.zeros(len(P) - DS[0] + 1)
        for i in range(len(P) - DS[0] + 1):
            # s_ = 0
            # for j in range(DS[i]):
            #     s_ += P[i + DS[0] - 1 - j:i + DS[0]].mean()
            # EP[i] = s_
            EP[i] = sum([P[i + DS[0] - 1 - j:i + DS[0]].mean() for j in range(DS[i])])
        return EP
    else:
        EP = numpy.zeros(len(P) - DS + 1)
        for i in range(len(P) - DS + 1):
            # s_ = 0
            # for j in range(DS):
            #     s_ += P[i + DS - 1 - j:i + DS].mean()
            # EP[i] = s_
            EP[i] = sum([P[i + DS - 1 - j:i + DS].mean() for j in range(DS)])
        return EP


# 已验证


# ----------------------------------------------------------使用生成器方法代替，提高效率, 但发现效率差不多，效率修正前低，修正后高
# def EP_generator(P, DS, DS0, i=0):
#     '''定义生成器计算EP'''
#     while True:
#         s = 0
#         for j in range(DS):
#             # 为了让变DS可能，这个DS_range要变化而下方的DS不能变，每次更改DS而不更改DS0
#             s += P[DS0 + i - j - 1: DS0 + i].mean()
#         [i, DS] = yield s
#
#
# def fun_EP(P, DS=365):
#     '''使用生成器计算EP'''
#     if type(DS) is numpy.ndarray:
#         EP = numpy.zeros(len(P) - DS[0] + 1)
#         EP_ = EP_generator(P, DS[0], DS[0])
#         EP[0] = next(EP_)
#         for i in range(1, len(P) - DS[0] + 1):
#             EP[i] = EP_.send([i, DS[i]])
#         return EP
#         EP_.close()
#     else:
#         EP_ = EP_generator(P, DS, DS)
#         EP = numpy.zeros(len(P) - DS + 1)
#         EP[0] = next(EP_)
#         for i in range(1, len(P) - DS + 1):
#             EP[i] = EP_.send([i, DS])
#         return EP
#         EP_.close()
# ----------------------------------------------------------------

def ac_P(P, DS=365):
    '''计算累积降水、多年平均降水、累积降水亏损'''
    if type(DS) is numpy.ndarray:
        # p_sum = numpy.zeros(len(P) - DS[0] + 1)
        # for i in range(len(P) - DS[0] + 1):
        #     p_sum[i] = P[i + DS[0] - DS[i]:i + DS[0]].sum()
        # 用列表推导式代替，速度快
        p_sum = numpy.array([P[i + DS[0] - DS[i]:i + DS[0]].sum() for i in range(len(P) - DS[0] + 1)])
        # n = math.floor(len(p_sum) / 365)
        n = len(p_sum) // 365
        p_sum1 = p_sum[:n * 365].reshape((n, 365))
        AVG = p_sum1.mean(0)
        APD = p_sum1 - AVG
        APD = APD.reshape((n * 365,))
        return p_sum, AVG, APD
    else:
        # p_sum = numpy.zeros(len(P) - DS + 1)
        # for i in range(len(P) - DS + 1):
        #     p_sum[i] = P[i:i + DS].sum()
        p_sum = numpy.array([P[i:i + DS].sum() for i in range(len(P) - DS + 1)])
        # n = math.floor(len(p_sum) / 365)
        n = len(p_sum) // 365
        p_sum1 = p_sum[:n * 365].reshape((n, 365))
        AVG = p_sum1.mean(0)
        APD = p_sum1 - AVG
        APD = APD.reshape((n * 365,))
        return p_sum, AVG, APD


# 已验证


def move_average(MEP, d=5):
    # for i in range(len(MEP) - 2 * math.floor(d / 2)):
    for i in range(len(MEP) - 2 * (d // 2)):
        # MEP[i + math.floor(d / 2)] = MEP[i:i + d].mean()
        MEP[i + d // 2] = MEP[i:i + d].mean()
    return MEP


# 已验证


def stastic(EP):
    '''计算EP的统计参数，均值，亏损，亏损标准化'''
    # n = math.floor(len(EP) / 365)
    n = len(EP) // 365
    EP = EP[:n * 365]
    EP = EP.reshape((n, 365))
    MEP = EP.mean(0)
    MEP = move_average(MEP)
    DEP = EP - MEP
    SEP = DEP / EP.std(0, ddof=1)
    EDI = DEP / DEP.std(0, ddof=1)
    # reshape
    DEP = DEP.reshape((n * 365,))
    SEP = SEP.reshape((n * 365,))
    EDI = EDI.reshape((n * 365,))
    return n, MEP, DEP, SEP, EDI


# TODO 对时间进行处理对应，已验证


def fun_PRN(DEP, DS=365):
    '''PRN计算'''
    if type(DS) is numpy.ndarray:
        wight_array = numpy.zeros(len(DEP))
        # (len(DEP) - DS[0] + 1) - (len(DEP) - DS[0] + 1) % 365
        for i in range(len(wight_array)):
            wight_array[i] = sum([1 / (j + 1) for j in range(DS[i])])
            # wight = 0
            # for j in range(DS[i]):
            #     wight += 1 / (j + 1)
            # wight_array[i] = wight
        PRN = DEP / wight_array
        return PRN
    else:
        # wight = 0
        wight_array = numpy.zeros(len(DEP))  # 去尾，为了与DEP对应，才能相除
        # (len(DEP) - DS + 1) - (len(DEP) - DS + 1) % 365
        # for i in range(DS):
        #     wight += 1 / (i + 1)
        wight = sum([1 / (i + 1) for i in range(DS)])
        wight_array[:] = wight
        PRN = DEP / wight
        return PRN


# 已验证


def DS_cal(P, SEP, h=0, DS=365):
    '''基于SEP的DS计算'''
    CNS, ANES, flag_start, flag_end = run_theory(SEP, h)
    m = len(CNS)
    DS_array = numpy.zeros((len(P) - DS + 1,))
    DS_array = DS
    DR_array = numpy.zeros((len(P) - DS + 1,))
    for i in range(m):
        DR_array[flag_start[i]:flag_end[i] + 1] = numpy.array((range(CNS[i])))
    DS_array = (DS_array + DR_array).astype('int')
    return DS_array


# 已验证


def DS_modify(P, DS_array):
    '''DS修正'''
    EP = fun_EP(P, DS_array)  # 已验证
    n, MEP, DEP, SEP, EDI = stastic(EP)  # 不需验证
    p_sum, AVG, APD = ac_P(P, DS_array)  # 已验证
    PRN = fun_PRN(DEP, DS_array)  # 已验证
    return EP, n, MEP, DEP, SEP, EDI, p_sum, AVG, APD, PRN


# 已验证


def data_write(x, filename):
    if type(x) is pandas.core.frame.DataFrame:
        writer = pandas.ExcelWriter(filename)
        x.to_excel(writer, float_format='%.3f')
        writer.save()
        writer.close()
    else:
        test_data = pandas.DataFrame(x)
        writer = pandas.ExcelWriter(filename)
        test_data.to_excel(writer, float_format='%.3f')
        writer.save()
        writer.close()


def data_read(file_name):
    x = pandas.read_excel(file_name)
    x = x.values.reshape((x.values.shape[0],))
    return x


def create_Dataframe(start, end, P, EP, n, MEP, DEP, SEP, EDI, p_sum, AVG, APD, PRN, DS=365, freq='D'):
    '''构建数据组'''
    Dstart = datetime.date(start[0], start[1], start[2])  # start=[year, month, day]
    Dend = datetime.date(end[0], end[1], end[2])  # end=[year, month, day]
    m = len(EP)
    # 构建P的dataframe
    date_P = pandas.date_range(start=Dstart, end=Dend, freq=freq)
    P = pandas.DataFrame(P, index=date_P, columns=['P'])
    # 构建EP, p_sum的dataframe(长度相同)
    Dstart = Dstart + datetime.timedelta(days=DS - 1)  # 往后延DS-1个
    date_EP = pandas.date_range(start=Dstart, end=Dend, freq=freq)
    date_p_sum = date_EP
    EP = pandas.DataFrame(EP, index=date_EP, columns=['EP'])
    p_sum = pandas.DataFrame(p_sum, index=date_p_sum, columns=['p_sum'])
    # 构建MEP，AVG的dataframe
    MEP = numpy.hstack((numpy.tile(MEP, [n, ]), MEP[:m % 365]))  # 错误使用MEP.repeat(n)
    AVG = numpy.hstack((numpy.tile(AVG, [n, ]), AVG[:m % 365]))
    date_MEP = date_EP
    date_AVG = date_EP
    MEP = pandas.DataFrame(MEP, index=date_MEP, columns=['MEP'])
    AVG = pandas.DataFrame(AVG, index=date_AVG, columns=['AVG'])
    # 构建DEP,SEP,EDI,APD,PRN的dataframe
    Dend = Dend - datetime.timedelta(days=m % 365)  # 往前len(EP)(m)%365个(DS表示时间段，所以-1，len(EP)%365表示时间点)
    date_DEP = pandas.date_range(start=Dstart, end=Dend, freq=freq)
    date_SEP = date_DEP
    date_EDI = date_DEP
    date_APD = date_DEP
    date_PRN = date_DEP
    DEP = pandas.DataFrame(DEP, index=date_DEP, columns=['DEP'])
    SEP = pandas.DataFrame(SEP, index=date_SEP, columns=['SEP'])
    EDI = pandas.DataFrame(EDI, index=date_EDI, columns=['EDI'])
    APD = pandas.DataFrame(APD, index=date_APD, columns=['APD'])
    PRN = pandas.DataFrame(PRN, index=date_PRN, columns=['PRN'])
    # 基于日期链接
    P = P.join(EP)
    P = P.join(MEP)
    P = P.join(DEP)
    P = P.join(SEP)
    P = P.join(EDI)
    P = P.join(p_sum)
    P = P.join(AVG)
    P = P.join(APD)
    P = P.join(PRN)
    return P


def plot_EDI(data):
    plt.figure()  # dpi=450
    # 图1 降水
    plt.subplot(3, 1, 1)
    plt.bar(data['P'].index, data['P'].values, facecolor='#63B8FF', label='P')
    plt.legend()
    # 图2 MEP,EP,SEP*100,0
    plt.subplot(3, 1, 2)
    plt.plot(data['MEP'].index, data['MEP'].values, color='grey', linestyle='-', label='MEP', linewidth=0.8)
    plt.plot(data['EP'].index, data['EP'].values, color='#1874CD', linestyle=':', label='EP')
    plt.plot(data['SEP'].index, 100 * data['SEP'].values, color='#104E8B', linestyle='--', label='100*SEP',
             linewidth=0.8)
    plt.plot([data['P'].index[0].date(), data['P'].index[-1].date()], [0, 0], color='#B3B3B3', linestyle='-', label='0',
             linewidth=0.8)
    plt.legend()
    # 图3 EDI*100 ANES APD PRN*10
    plt.subplot(3, 1, 3)
    plt.plot(data['EDI'].index, 100 * data['EDI'].values, color='grey', linestyle='-', label='100*EDI', linewidth=0.8)
    plt.plot(data['APD'].index, data['APD'].values, color='#104E8B', linestyle='--', label='APD', linewidth=0.8)
    plt.plot(data['PRN'].index, 10 * data['PRN'].values, color='#104E8B', linestyle='-.', label='10*PRN', linewidth=0.8)
    plt.plot([data['P'].index[0].date(), data['P'].index[-1].date()], [0, 0], color='#B3B3B3', linestyle='-', label='0',
             linewidth=0.8)
    plt.legend()
    # TODO ANES没算没画,颜色调一下


def EDI(P, start, end, h_sep=0, DS=365, freq='D'):  # file_name_p
    # P = data_read(file_name_p)
    EP = fun_EP(P, DS)
    p_sum, AVG, APD = ac_P(P, DS)
    n, MEP, DEP, SEP, EDI = stastic(EP)
    PRN = fun_PRN(DEP, DS)
    DS_array = DS_cal(P, SEP, h_sep, DS)
    EP, n, MEP, DEP, SEP, EDI, p_sum, AVG, APD, PRN = DS_modify(P, DS_array)
    # start = [1960, 1, 1]
    # end = [2012, 7, 31]
    data = create_Dataframe(start, end, P, EP, n, MEP, DEP, SEP, EDI, p_sum, AVG, APD, PRN, DS, freq)
    # data_write(data, 'data.xlsx')
    # plot_EDI(data)
    return data


def EDI_only(P, h_sep=0, DS=365):
    EP = fun_EP(P, DS)
    n, MEP, DEP, SEP, EDI = stastic(EP)
    DS_array = DS_cal(P, SEP, h_sep, DS)
    _, _, _, _, _, EDI, *_ = DS_modify(P, DS_array)
    return EDI


if __name__ == "__main__":
    # starttime1 = time.time()
    P = data_read('results/P_test.xlsx')
    EDI = EDI_only(P, h_sep=0, DS=365)
    # endtime1 = time.time()
    # dtime1 = endtime1 - starttime1
    # print(dtime1)
