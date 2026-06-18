import openpyxl

import config

'''
单井对比先运行GA再DE再PSO再混合
'''
import read_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from sko.GA import GA
from sko.PSO import PSO
from sko.tools import set_run_mode
from config import *
from openpyxl import load_workbook

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


    set_run_mode(func, 'multiprocessing')


    ga = GA(func=func, n_dim=3, size_pop=50, max_iter=150,
            lb=lb, ub=ub)
    ga.run()
    # The data is closest to the paper results once：C=26.88, epsilon=0.10846, gamma=0.0298



    print('best_x is ', ga.best_x, 'best_y is', ga.best_y)
    gbest = ga.best_x
    # plot the results
    GA_pred = show_result(gbest, Y_test)



    _, _, _, Y_test,_ = read_data.prepare_dataforSingle(data, pots[i], feature_list)

    mape = mean_absolute_percentage_error(Y_test[:], GA_pred[:])
    rmse = mean_squared_error(Y_test[:], GA_pred[:]) ** 0.5
    mae = mean_absolute_error(Y_test[:], GA_pred[:])
    print(pots[i],"评价指标（GA优化）：")
    print('mape: ', mape)
    print('rmse:', rmse)
    print('mae:', mae)


    #Excel写入
    savedata1 = {'Y_test': Y_test,
                'GA_pred': GA_pred,

                }
    savedata2 = [[pots[i]+' ga.best_x', ga.best_x[0], ga.best_x[1], ga.best_x[2]]]
    savedata3 = [[pots[i]+' GAmape',mape],
                [pots[i]+' GArmse',rmse],
                [pots[i]+' GAmae',mae]]



    savedata1 = pd.DataFrame(savedata1)

    with pd.ExcelWriter(config.ResultPath,mode='a') as writer:
        savedata1.to_excel(writer, sheet_name=pots[i], float_format='%.5f')
        # savedata2.to_excel(writer, sheet_name=pots[i], float_format='%.5f')
        # savedata3.to_excel(writer, sheet_name=pots[i], float_format='%.5f')

    wb = openpyxl.load_workbook(config.ResultPath)
    wa = wb.active
    for i in savedata2:
        wa.append(i)
    for j in savedata3:
        wa.append(j)

    wb.save(config.ResultPath)





