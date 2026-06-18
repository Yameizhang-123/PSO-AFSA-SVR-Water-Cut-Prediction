'''
使用SVR进行含水率预测，不使用优化算法
'''
import openpyxl

import config
import read_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from sko.DE import DE
from sko.tools import set_run_mode
from config import *



data = pd.read_csv('庄2油井月度生产数据.csv', encoding='utf-8')



for i in range(0, len(pots)):
    X_train, Y_train, X_test, Y_test, _ = read_data.prepare_dataforSingle(data, pots[i], feature_list)

    def show_result(gBest, Y_test):
        x1, x2, x3 = gBest
        model_svr = SVR(C=x1, epsilon=x2, gamma=x3)
        model_svr.fit(X_train, Y_train)
        Y_pred = model_svr.predict(X_test)
        plt.plot(Y_pred, label='Y_pred')
        plt.plot(Y_test, label='Y_test')
        plt.legend()
        plt.show()
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





    _, _, _, Y_test,_ = read_data.prepare_dataforSingle(data, pots[i], feature_list)

    mape = mean_absolute_percentage_error(Y_test[:], Y_pred[:])
    rmse = mean_squared_error(Y_test[:], Y_pred[:]) ** 0.5
    mae = mean_absolute_error(Y_test[:], Y_pred[:])
    print(pots[i],"评价指标（SVR无优化）：")
    print('mape: ', mape)
    print('rmse:', rmse)
    print('mae:', mae)

    # Excel写入
    # savedata1 = {'Y_test': Y_test,
    #              'DE_pred': DE_pred,
    #              }
print('exp over')

