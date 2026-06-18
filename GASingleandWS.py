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

        #画图显示
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


    # GA模块



    set_run_mode(func, 'multiprocessing')

    ga = GA(func=func, n_dim=3, size_pop=50, max_iter=150,
            lb=lb, ub=ub)
    ga.run()
    # The data is closest to the paper results once：C=26.88, epsilon=0.10846, gamma=0.0298


    #print(pots[i],'GA Running time: %s Seconds' % (GAend - GAstart))


    print('best_x is ', ga.best_x, 'best_y is', ga.best_y)
    GAgbest = ga.best_x
    # plot the results
    GA_pred = show_result(GAgbest, Y_test)
    #GA模块结束
    '''
     # GA模块
    GAstart = time.time()


    set_run_mode(func, 'multiprocessing')

    ga = GA(func=func, n_dim=3, size_pop=50, max_iter=150,
            lb=lb, ub=ub)
    ga.run()
    # The data is closest to the paper results once：C=26.88, epsilon=0.10846, gamma=0.0298

    GAend = time.time()
    print(GAend - GAstart)
    #print(pots[i],'GA Running time: %s Seconds' % (GAend - GAstart))


    print('best_x is ', ga.best_x, 'best_y is', ga.best_y)
    GAgbest = ga.best_x
    # plot the results
    GA_pred = show_result(GAgbest, Y_test)
    #GA模块结束
    
    
    #DE模块

    de = DE(func=func, n_dim=3, size_pop=50, max_iter=150,
            lb=lb, ub=ub,
            prob_mut=0.3)
    de.run()


    #print( pots[i],'DE Running time: %s Seconds' % (DEend - DEstart))
    print('best_x is ', de.best_x, 'best_y is', de.best_y)
    DEgbest = de.best_x
    # plot the results
    DE_pred = show_result(DEgbest, Y_test)
    #DE模块结束



    #PSO模块

    pso = PSO(func=func, n_dim=3, pop=40, max_iter=150,
              lb=lb, ub=ub,
              w=0.8, c1=0.5, c2=0.5, verbose=False)#verbose是否需要进度条
    pso.run()


    #print(pots[i],'PSO Running time: %s Seconds' % (PSOend - PSOstart))
    # The data is closest to the paper results once：C=26.88, epsilon=0.10846, gamma=0.0298
    print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    PSOgbest = pso.gbest_x
    # plot the results
    PSO_pred = show_result(PSOgbest, Y_test)
    # PSO模块结束

    #AFSA

    afsa = AFSA_sko(func, n_dim=3, size_pop=50, max_iter=50,
                    max_try_num=100, step=0.5, visual=0.3,
                    q=0.98, delta=0.5, X=PSOgbest)
    AFSAgbest, best_y = afsa.run()

    #print(pots[i],'AFSA Running time: %s Seconds' % (AFSAend - AFSAstart))

    print(AFSAgbest, best_y)
    AFSA_pred = show_result(AFSAgbest, Y_test)
    #AFSA模块结束
    '''
    #时间滑窗
    dates = Y_date

    results = []
    y_real = []
    window_size = 6 #窗口大小 可修改
    count = 0
    temp = window_size

    while temp > 0:
        results.extend([-1])
        y_real.extend([-1])
        temp = temp - 1

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
        model_svr = SVR(C=GAgbest[0], epsilon=GAgbest[1], gamma=GAgbest[2])

        model_svr.fit(X_train, Y_train)
        WS_pred = model_svr.predict(X_test)
        results.extend(WS_pred)
        y_real.extend(Y_test)
    #时间滑窗结束



    #计算评价指标
    _, _, _, Y_test,_ = read_data.prepare_dataforSingle(data, pots[i], feature_list)

    GAmape = mean_absolute_percentage_error(Y_test[:], GA_pred[:])
    GArmse = mean_squared_error(Y_test[:], GA_pred[:]) ** 0.5
    GAmae = mean_absolute_error(Y_test[:], GA_pred[:])

    '''    
    DEmape = mean_absolute_percentage_error(Y_test[:], DE_pred[:])
    DErmse = mean_squared_error(Y_test[:], DE_pred[:]) ** 0.5
    DEmae = mean_absolute_error(Y_test[:], DE_pred[:])

    PSOmape = mean_absolute_percentage_error(Y_test[:], PSO_pred[:])
    PSOrmse = mean_squared_error(Y_test[:], PSO_pred[:]) ** 0.5
    PSOmae = mean_absolute_error(Y_test[:], PSO_pred[:])

    AFSAmape = mean_absolute_percentage_error(Y_test[:], AFSA_pred[:])
    AFSArmse = mean_squared_error(Y_test[:], AFSA_pred[:]) ** 0.5
    AFSAmae = mean_absolute_error(Y_test[:], AFSA_pred[:])'''

    WSmape = mean_absolute_percentage_error(y_real[window_size:], results[window_size:])
    WSrmse = mean_squared_error(y_real[window_size:], results[window_size:]) ** 0.5
    WSmae = mean_absolute_error(y_real[window_size:], results[window_size:])

    print('GAmape: ', GAmape)
    print('GArmse:', GArmse)
    print('GAmae:', GAmae)

    print('WSmape: ', WSmape)
    print('WSrmse:', WSrmse)
    print('WSmae:', WSmae)
    print('over')

