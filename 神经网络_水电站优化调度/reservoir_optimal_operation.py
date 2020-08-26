# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.stats import gamma
import tensorflow as tf
from tensorflow import keras

# 数据读取
ZV = pd.read_excel('F:/小论文2/代码/神经网络_水电站优化调度/data.xlsx', sheet_name=0)
ZQ = pd.read_excel('F:/小论文2/代码/神经网络_水电站优化调度/data.xlsx', sheet_name=1)
Q = pd.read_excel('F:/小论文2/代码/神经网络_水电站优化调度/data.xlsx', sheet_name=2)

# 拟合曲线 zsy-v, zxy-qfd
zv, residuals_zv, *_ = np.polyfit(ZV['Z'], ZV['V'], deg=3, full=True)  # 拟合效果好，直接调用函数 TODO 待调用 zv曲线 np.polyval(zv, x_zq)
vz = np.polyfit(ZV['V'], ZV['Z'], deg=3)
x_zv = np.linspace(ZV['Z'].min(), ZV['Z'].max(), 100)
zq, residuals_zq, *_ = np.polyfit(ZQ['Z'], ZQ['Q'], deg=3, full=True)  # 拟合效果好，直接调用函数 TODO 待调用 zq曲线 np.polyval(zq, x_zq)
qz = np.polyfit(ZQ['Q'], ZQ['Z'], deg=3)  # TODO 待调用 qz曲线 np.polyval(qz, x_zq)
x_zq = np.linspace(ZQ['Z'].min(), ZQ['Z'].max(), 100)

# 插值
# zv_inter = interpolate.interp1d(ZV['Z'], ZV['V'])  # TODO 待调用 zv曲线 zv_inter(x_zv)
# vz_inter = interpolate.interp1d(ZV['V'], ZV['Z'])  # TODO 待调用 vz曲线 vz_inter(x_zv)

# 绘图显示
plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(ZV['Z'], ZV['V'])
plt.plot(x_zv, np.polyval(zv, x_zv), label='polyfit')
# plt.plot(x_zv, zv_inter(x_zv), linestyle='-.', label='interpolate')
plt.xlabel("Z_sy")
plt.ylabel("V")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x_zq, np.polyval(zq, x_zq), label='polyfit')
plt.scatter(ZQ['Z'], ZQ['Q'])
plt.xlabel("Z_xy")
plt.ylabel("qfd")
plt.legend()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

# 初始参数
step = 12  # 阶段数
ziter = int(1)  # 离散精度
z_str = 685  # 初水位，死水位
z_end = 685  # 末水位，死水位
z_max = 704  # 水位上限，正常蓄水位
z_min = 685  # 水位下限，死水位
z_678max = 695  # 678月水位上限
k = 8.5  # 系数
N_min = 78  # 最小出力限制，保证出力
N_max = 300  # 最大出力限制，装机容量
UB = np.ones((13,))  # 各月水位上限
UB[:] = z_max
UB[0] = z_str
UB[-1] = z_end
UB[2:6] = z_678max  # 678月水位限制,考虑初末时段
LB = np.ones((13,))  # 各月水位下限
LB[:] = z_min
LB[0] = z_str
LB[-1] = z_end
qrk = Q.values.flatten()


def grade_(start, end, step_):
    x_ = [int(x) for x in np.linspace(start, end, int((end - start) / step_))]
    if start == end:  # 为了满足后面主程序需求
        x_ = [start]
    return x_


def qfd(z_sy1, z_sy2, qrk):
    return qrk + (np.polyval(zv, z_sy1) - np.polyval(zv, z_sy2))*(10**8)/(30*24*3600)


def h(z_sy1, z_sy2, qfd):
    return (z_sy1 + z_sy2) / 2 - np.polyval(qz, qfd)


def N(k, qfd, h):
    return k * qfd * h / 1000


def output(z_sy1, z_sy2, qrk):
    Qele = qfd(z_sy1, z_sy2, qrk)  # 发电流量
    Hele = h(z_sy1, z_sy2, qfd(z_sy1, z_sy2, qrk))  # 水头
    Nele = N(k, Qele, Hele)  # 出力
    Qdis = 0
    if Nele > 300:
        dNele = Nele - 300
        Nele = 300
        Qdis = dNele/k/Hele*1000
    return Qele, Qdis, Hele, Nele


# 动态规划寻优
def solution(z_min, z_max, ziter, step, LB, UB, k, N_max, N_min, z_str, z_end, qrk):
    Z_linespace = grade_(z_min, z_max, ziter)
    statNum = len(Z_linespace)  # 状态变量
    benefit = -1 * np.ones((statNum, step))  # 效益矩阵（发电量）,19（离散状态）*12(阶段)
    index = np.zeros((statNum, step))  # 最大效益的index，用于储存水位
    for i in range(step-1, -1, -1):
        Z_1 = grade_(LB[i], UB[i], ziter)  # i时段差分
        Z_2 = grade_(LB[i + 1], UB[i + 1], ziter)  # i+1时段差分,包含决策
        value = np.zeros((len(Z_1), len(Z_2)))
        temp = np.zeros((len(Z_1), len(Z_2)))
        if i == step - 1:  # 最后一个阶段，i == step - 1
            for j in range(len(Z_1)):
                z_sy1 = Z_1[j]  # 时段初
                z_sy2 = z_end  # 时段末,不用决策，初始状态
                k_ = 0
                value[j, k_] = N(k, qfd(z_sy1, z_sy2, qrk[i]), h(z_sy1, z_sy2, qfd(z_sy1, z_sy2, qrk[i])))
                if value[j, k_] > N_max:
                    value[j, k_] = N_max
                if value[j, k_] < N_min:
                    value[j, k_] = -100000000000
                benefit[j, i] = value[j, k_]
        if i < step - 1:
            for j in range(len(Z_1)):
                for k_ in range(len(Z_2)):
                    z_sy1 = Z_1[j]
                    z_sy2 = Z_2[k_]
                    value[j, k_] = N(k, qfd(z_sy1, z_sy2, qrk[i]), h(z_sy1, z_sy2, qfd(z_sy1, z_sy2, qrk[i])))
                    if value[j, k_] > N_max:
                        value[j, k_] = N_max
                    if value[j, k_] < N_min:
                        value[j, k_] = -100000000000
                    temp[j, k_] = value[j, k_] + benefit[k_, i + 1]
                benefit[j, i] = np.max(temp[j, :])
                index[j, i] = np.argmax(temp[j, :])

    # 反演最优轨迹
    state = np.zeros((step + 1,))
    opt_z = np.zeros((step + 1,))
    state[0] = np.argmax(benefit[:, 0])
    opt_z[0] = LB[0] + ziter * (state[0])
    opt_z[-1] = z_end
    for i in range(step - 1):
        Z_2 = grade_(LB[i + 1], UB[i + 1], ziter)  # i+1时段差分,包含决策
        state[i + 1] = index[int(state[i]), i]
        # opt_z[i + 1] = LB[i + 1] + ziter * (state[i + 1] + 1)
        opt_z[i+1] = Z_2[int(state[i + 1])]

    # 提取调度结果
    # opt_z = solution(z_min, z_max, ziter, step, LB, UB, k, N_max, N_min, z_str, z_end, qrk)
    result = np.zeros((step, 6))
    result[:, 0] = qrk
    for i in range(step):
        z_sy1 = opt_z[i]
        z_sy2 = opt_z[i + 1]
        Qele, Qdis, Hele, Nele = output(z_sy1, z_sy2, qrk[i])
        result[i, 1] = Qele
        result[i, 2] = Qdis
        result[i, 3] = Hele
        result[i, 4] = Nele
    result[:, 5] = opt_z[1:]
    # result = pd.DataFrame(result, columns=['qrk', 'Qele', 'Qdis', 'Hele', 'Nele', 'opt_z'])
    return result[:, 4]

    # 绘图
    # plt.figure()
    # plt.plot(list(range(step+1)), opt_z)
    # plt.xlabel("month")
    # plt.ylabel("Z")

# # 生成随机样本
# sample = np.zeros((1000, 12))
# for i in range(12):
#     sample[:, i] = gamma.rvs(size=1000, a=qrk[i])
#
# # 对样本进行动态规划
# N_sample = np.zeros((1000, 12))
# for i in range(1000):
#     N_ = solution(z_min, z_max, ziter, step, LB, UB, k, N_max, N_min, z_str, z_end, sample[i, :])
#     N_sample[i, :] = N_
#
# # 数据存储
# np.savetxt('输入样本.txt', sample)
# np.savetxt('输出样本.txt', N_sample)

# 搭建神经网络进行训练
sample = np.loadtxt('输入样本.txt')
N_sample = np.loadtxt('输出样本.txt')
x_train = sample[:700, :]  # 训练集700
y_train = N_sample[:700, :]  # 验证集300
x_test = sample[700:, :]
y_test = N_sample[700:, :]
model = tf.keras.Sequential([tf.keras.layers.Dense(8, input_shape=(12, ), activation='relu',),
                             tf.keras.layers.Dense(12)])
model.compile(optimizer='adam', loss='mse')
history = model.fit(x_train, y_train, epochs=100)
plt.figure()
plt.plot(history.epoch, history.history.get('loss'))
plt.xlabel("epochs")
plt.ylabel("loss")
evalute_result = model.evaluate(x_test, y_test)

# 优化预测模型


# 预测并绘制预测比较图
# def month(y, step):
#     mon = np.zeros((y.shape[0], step+1))
#     for i in range(y.shape[0]):
#         mon[i, :] = list(range(step+1))
#     return mon
qrk_predict = x_test
opt_z_predict = model.predict(x_test)
plt.figure()
cx_values = list(range(x_test.shape[0]))
cy_values = [0.2/x_test.shape[0]*x for x in cx_values]
for i in range(x_test.shape[0]):
    if i == x_test.shape[0]-1:
        plt.plot(list(range(step)), opt_z_predict[i, :], label='predict',
                 alpha=cy_values[i],
                 color='deepskyblue')  # alpha=cy_values[i], color='deepskyblue' , linestyle='--', linewidth=0.8,
        plt.scatter(list(range(step)), y_test[i, :], label='real', alpha=cy_values[i] + 0.2,
                    color='royalblue')  # , alpha=cy_values[i], color='violet'
    else:
        plt.plot(list(range(step)), opt_z_predict[i, :],
                 alpha=cy_values[i], color='deepskyblue')  # alpha=cy_values[i], color='deepskyblue' , linestyle='--', linewidth=0.8,
        plt.scatter(list(range(step)), y_test[i, :], alpha=cy_values[i]+0.2, color='royalblue')  # , alpha=cy_values[i], color='violet'
plt.legend()
plt.xlabel('month')
plt.ylabel('N')

N_zsn_pre = model.predict([[qrk], [qrk]])  # 因为不能预测1维的
# 模型输出并存储
model.save('my_model.h5')
# 模型读取
model = keras.models.load_model('my_model.h5')
