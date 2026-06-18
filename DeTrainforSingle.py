'''
使用DE优化方法结合SVR进行含水率预测
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


    set_run_mode(func, 'multiprocessing')

    de = DE(func=func, n_dim=3, size_pop=50, max_iter=150,
            lb=lb, ub=ub,
            prob_mut=0.3)
    de.run()

    print('best_x is ', de.best_x, 'best_y is', de.best_y)
    gbest = de.best_x
    # plot the results
    DE_pred = show_result(gbest, Y_test)




    _, _, _, Y_test,_ = read_data.prepare_dataforSingle(data, pots[i], feature_list)

    mape = mean_absolute_percentage_error(Y_test[:], DE_pred[:])
    rmse = mean_squared_error(Y_test[:], DE_pred[:]) ** 0.5
    mae = mean_absolute_error(Y_test[:], DE_pred[:])
    print(pots[i],"评价指标（DE优化）：")
    print('mape: ', mape)
    print('rmse:', rmse)
    print('mae:', mae)

    # Excel写入
    # savedata1 = {'Y_test': Y_test,
    #              'DE_pred': DE_pred,
    #              }
    savedata1 = [[pots[i] +'Y_test', Y_test],
                 [pots[i] +'DE_pred', DE_pred]]
    savedata2 = [[pots[i] + ' de.best_x', de.best_x[0], de.best_x[1], de.best_x[2]]]
    savedata3 = [[pots[i] + ' DEmape', mape],
                 [pots[i] + ' DErmse', rmse],
                 [pots[i] + ' DEmae', mae]]

    # savedata1 = pd.DataFrame(savedata1)
    #
    # with pd.ExcelWriter(config.ResultPath, mode='a') as writer:
    #     savedata1.to_excel(writer, sheet_name=pots[i], float_format='%.5f')
    #     # savedata2.to_excel(writer, sheet_name=pots[i], float_format='%.5f')
    #     # savedata3.to_excel(writer, sheet_name=pots[i], float_format='%.5f')

    wb = openpyxl.load_workbook(config.ResultPath)

    wa = wb['Sheet1']
    for i in savedata2:
        wa.append(i)
    for j in savedata3:
        wa.append(j)

    wb.save(config.ResultPath)


