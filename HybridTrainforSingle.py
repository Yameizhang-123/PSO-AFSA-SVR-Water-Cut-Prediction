'''
使用PSO和AFSA结合的优化方法结合SVR'对单口井'进行含水率预测
用A井（可替换）7/10的数据作为训练集获得SVR三个参数，用在预测A井3/10的训练数据上
指标为MAPE
'''


import read_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.utils
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
from AFSA_sko import AFSA_sko
from sko.PSO import PSO
from sko.tools import set_run_mode
from config import *

data = pd.read_csv('庄2油井月度生产数据.csv', encoding='utf-8')

for i in range(0, len(pots)):
    X_train, Y_train, X_test, Y_test, Y_date = read_data.prepare_dataforSingle(data, pots[i], feature_list)


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

    pso = PSO(func=func, n_dim=3, pop=40, max_iter=150,
              lb=lb, ub=ub,
              w=0.8, c1=0.5, c2=0.5, verbose=True)
    pso.run()
    # The data is closest to the paper results once：C=26.88, epsilon=0.10846, gamma=0.0298
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    gbest = pso.gbest_x
    # plot the results
    show_result(gbest, Y_test)

    # gbest = [9.69549198e+00, 6.00685147e-03, 3.75263213e-04]

    afsa = AFSA_sko(func, n_dim=3, size_pop=50, max_iter=1,
                    max_try_num=100, step=0.5, visual=0.3,
                    q=0.98, delta=0.5, X=gbest)
    gbest, best_y = afsa.run()
    print(gbest, best_y)
    AFSAY_pred = show_result(gbest, Y_test)



    dates = Y_date

    results = []
    y_real = []
    window_size = 2
    count = 0

    for date in dates[window_size:]:
        date = int(date)
        count += 1
        # print(date)

        X_train, Y_train, X_test, Y_test, _= read_data.prepare_dataforSingle(data, pots[i], feature_list,
                                                        time_window_start=int(date) - window_size - 1,
                                                        time_window_end=int(date),
                                                        clip_zero=False, isWindow=True)

        # gbest = [9.69549198e+00, 6.00685147e-03, 3.75263213e-04]
        model_svr = SVR(C=gbest[0], epsilon=gbest[1], gamma=gbest[2])

        model_svr.fit(X_train, Y_train)
        Y_pred = model_svr.predict(X_test)
        results.extend(Y_pred)
        y_real.extend(Y_test)

    #_, _, _, Y_test = read_data.prepare_dataforSingle(data, pots[i], feature_list)
    plt.plot(results[:], label='Hybrid_Pre')
    plt.plot(y_real[:], label='Y_test')
    plt.legend()
    plt.show()

    print('mape: ', mean_absolute_percentage_error(y_real[:], results[:]))
    print('rmse:', mean_squared_error(y_real[:], results[:]) ** 0.5)
    print('mae:', mean_absolute_error(y_real[:], results[:]))
    print(count + ':mytrain end')