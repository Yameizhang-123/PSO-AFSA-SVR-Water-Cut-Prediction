import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def plot_all_pot(input_data):
    for i in input_data['井号'].unique():
        data_pot = input_data[input_data['井号'] == i]
        plt.plot(data_pot['含水'][data_pot['年月'] > 199600])
        plt.legend([i])
        plt.show()

def HeatMap_prepare_data(input_data, pot: list, feature: list, clip_zero=True):
    input_data = input_data[input_data['井号'].isin(pot)]
    if clip_zero:
        input_data = input_data[input_data['Water cut'] > 0]
    uniform_data = input_data[feature]

    return uniform_data




'''
对单口井的实验读数据：
    1.实现对一口井中的数据进行训练集与数据集的划分
    2.在这一口井中校验时间滑窗是否正确
'''
def prepare_dataforSingle(input_data, pot, feature: list, year_month=199600,
                 time_window_start=199600, time_window_end=201805,
                 clip_zero=True, normalize=True, shuffle=True, isWindow=False
                 ):
    if time_window_start % 100 > 12 or time_window_start % 100 < 1:
        time_window_start = time_window_start - time_window_start % 100 + 12 - 100

    if time_window_end % 100 > 12 or time_window_end % 100 < 1:
        time_window_end = time_window_end - time_window_end % 100 + 12 - 100

    '''

        if ((time_window_end % 100) % 12 == 0):
            time_window_end += 12
            time_window_end -= 100
        else:
            mon = time_window_end % 100  # start的月份
            time_window_end = time_window_end - mon + mon % 12
            time_window_end = time_window_end / 100 + 100 * (mon / 12)
    '''

    input_data = input_data[input_data['井号'] == pot]
    input_data = input_data[input_data['年月'] >= year_month]
    input_data = input_data[input_data['年月'] <= time_window_end]
    input_data = input_data[input_data['年月'] >= time_window_start]

    if clip_zero:
        input_data = input_data[input_data['含水'] > 0]

    if isWindow:
        x_train = input_data[feature][input_data['年月'] <= time_window_end - 1]  # pot的feature数据
        x_test = input_data[feature][input_data['年月'] == time_window_end]  # 新的时间内pot的feature数据
        y_train = input_data['含水'][input_data['年月'] <= time_window_end - 1]  # Pot的含水数据
        y_test = input_data['含水'][input_data['年月'] == time_window_end]  # 新的时间内pot的含水数据
        y_date = x_test['年月']  # 最终的时间条/无效

        # input_data to numpy
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_test = np.array(x_test)
        y_test = np.array(y_test)
    else:
        X, y = input_data[feature], input_data['含水']
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        y_date = x_test['年月']  # 最终的时间条

        # input_data to numpy
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()
        x_test = x_test.to_numpy()
        y_test = y_test.to_numpy()
        y_date = y_date.to_numpy()


    if normalize:
        # 标准化转换
        scaler = StandardScaler()
        # 训练标准化对象
        scaler.fit(x_train)
        # 转换数据集
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    if shuffle:
        # shuffle
        index = np.arange(x_train.shape[0])
        np.random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]

    return x_train, y_train, x_test, y_test, y_date


'''
对多口井的实验读数据：
    1.实现将所有井除待测井的数据作为训练集，待测井的数据作为测试集
    2.时间滑窗的时间校验
'''


def prepare_data(input_data, pot: list, feature: list, year_month=199600,
                 time_window_start=199600, time_window_end=201805,
                 clip_zero=True, normalize=True, shuffle=True
                 ):
    if time_window_start % 100 > 12 or time_window_start % 100 < 1:
        time_window_start = time_window_start - time_window_start % 100 + 12 - 100

    if time_window_end % 100 > 12 or time_window_end % 100 < 1:
        time_window_end = time_window_end - time_window_end % 100 + 12 - 100

    '''

        if ((time_window_end % 100) % 12 == 0):
            time_window_end += 12
            time_window_end -= 100
        else:
            mon = time_window_end % 100  # start的月份
            time_window_end = time_window_end - mon + mon % 12
            time_window_end = time_window_end / 100 + 100 * (mon / 12)
    '''
    pot.append('庄2')
    input_data = input_data[input_data['井号'].isin(pot)]
    input_data = input_data[input_data['年月'] >= year_month]
    input_data = input_data[input_data['年月'] <= time_window_end]
    input_data = input_data[input_data['年月'] >= time_window_start]

    if clip_zero:
        input_data = input_data[input_data['含水'] > 0]

    x_train = input_data[feature][input_data['井号'] != '庄2']  # 除庄2所有井的feature数据
    x_test = input_data[feature][input_data['井号'] == '庄2']  # 只有庄2的feature数据
    y_train = input_data['含水'][input_data['井号'] != '庄2']  # 除庄2所有井的含水数据
    y_test = input_data['含水'][input_data['井号'] == '庄2']  # 只有庄2的含水数据

    # input_data to numpy
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    if normalize:
        # 标准化转换
        scaler = StandardScaler()
        # 训练标准化对象
        scaler.fit(x_train)
        # 转换数据集
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    if shuffle:
        # shuffle
        index = np.arange(x_train.shape[0])
        np.random.shuffle(index)
        x_train = x_train[index]
        y_train = y_train[index]

    return x_train, y_train, x_test, y_test



def get_date(input_data, pot):
    data_pot = input_data[input_data['井号'] == pot]
    return data_pot['年月'][data_pot['年月'] > 199600]


def main():
    # Read the data
    data = pd.read_csv('庄2油井月度生产数据.csv', encoding='utf-8')

    # print(len(data['井号'].unique()))
    # print(data['井号'].unique())

    # plot_all_pot(data)

    # make list 2-10 2-11 2-13* 2-15 2-19 2-2 2-20 2-23 2-25 2-26
    # 2-29 2-3 2-32 2-34 2-35 2-36 2-39 2-40 2-43 2-45 2-46 2-47
    # 2-49 2-5 2-52 2-57 2-59 2-61 2-62 2-63 2-64

    data_label = ['庄2-10', '庄2-11', '庄2-13', '庄2-15', '庄2-19',
                  '庄2-2', '庄2-20', '庄2-23', '庄2-25', '庄2-26',
                  '庄2-29', '庄2-3', '庄2-32', '庄2-34', '庄2-35',
                  '庄2-36', '庄2-39', '庄2-40', '庄2-43', '庄2-45',
                  '庄2-46', '庄2-47', '庄2-49', '庄2-5', '庄2-52',
                  '庄2-57', '庄2-59', '庄2-61', '庄2-62', '庄2-63',
                  '庄2-64']

    feature_list = ['年月', '油层厚度', '泵径或油嘴', '泵效', '地层系数','生产天数']

    # prepared_data = prepare_data(data, data_label, feature_list)

    plot_all_pot(data[data['井号'].isin(data_label)])

    print()


if __name__ == '__main__':
    main()
