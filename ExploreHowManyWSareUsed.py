'''
模型SVR
改变时间滑窗大小
指标为MAPE
'''

import read_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from AFSA_sko import AFSA_sko
from sko.PSO import PSO
from sko.tools import set_run_mode
from config import *
from sko.DE import DE
from sko.GA import GA

data = pd.read_csv('庄2油井月度生产数据.csv', encoding='utf-8')

for i in range(0, len(pots)):
    X_train, Y_train, X_test, Y_test, Y_date = read_data.prepare_dataforSingle(data, pots[i], feature_list)


    def show_result(gBest, Y_test):
        x1, x2, x3 = gBest
        model_svr = SVR(C=x1, epsilon=x2, gamma=x3)
        model_svr.fit(X_train, Y_train)
        Y_pred = model_svr.predict(X_test)

        # 画图显示
        # plt.plot(Y_pred, label='Y_pred')
        # plt.plot(Y_test, label='Y_test')
        # plt.legend()
        # plt.show()
        return Y_pred


    def func(x):
        x1, x2, x3 = x
        model_svr = SVR(C=x1, epsilon=x2, gamma=x3)
        model_svr.fit(X_train, Y_train)
        Y_pred = model_svr.predict(X_test)
        # metrics.r2_score(Y_test, Y_pred)
        # return mean_squared_error(Y_test, Y_pred)

        return mean_absolute_percentage_error(Y_test, Y_pred)


    Svr = SVR(kernel='rbf', C=10, gamma=10, epsilon=0.03)
    Svr.fit(X_train, Y_train)
    Y_pred = Svr.predict(X_test)

    # 时间滑窗
    dates = Y_date

    results = []
    y_real = []
    window_size = 4 # 窗口大小 可修改
    count = 0
    # temp = window_size
    #
    # while temp > 0:
    #     results.extend([-1])
    #     y_real.extend([-1])
    #     temp = temp - 1

    for date in dates[window_size:]:
        date = int(date)
        count += 1
        # print(date)

        X_train, Y_train, X_test, Y_test, _ = read_data.prepare_dataforSingle(data, pots[i], feature_list,
                                                                              time_window_start=int(
                                                                                  date) - window_size - 1,
                                                                              time_window_end=int(date),
                                                                              clip_zero=False, isWindow=True)

        # gbest = [9.69549198e+00, 6.00685147e-03, 3.75263213e-04]
        # model_svr = SVR(C=AFSAgbest[0], epsilon=AFSAgbest[1], gamma=AFSAgbest[2])
        model_svr = SVR(C=10, gamma=10, epsilon=0.03)

        model_svr.fit(X_train, Y_train)
        WS_pred = model_svr.predict(X_test)
        results.extend(WS_pred)
        y_real.extend(Y_test)
    # 时间滑窗结束

    # 计算评价指标
    _, _, _, Y_test, _ = read_data.prepare_dataforSingle(data, pots[i], feature_list)

    GAmape = mean_absolute_percentage_error(Y_test[:], Y_pred[:])
    GArmse = mean_squared_error(Y_test[:], Y_pred[:]) ** 0.5
    GAmae = mean_absolute_error(Y_test[:], Y_pred[:])

    WSmape = mean_absolute_percentage_error(y_real[:], results[:])
    WSrmse = mean_squared_error(y_real[:], results[:]) ** 0.5
    WSmae = mean_absolute_error(y_real[:], results[:])

    print('mape: ', GAmape)
    print('rmse:', GArmse)
    print('mae:', GAmae)

    print('WSmape: ', WSmape)
    print('WSrmse:', WSrmse)
    print('WSmae:', WSmae)
    print('over')

