

# pots = ['庄2-10', '庄2-11', '庄2-13', '庄2-15', '庄2-19',
#         '庄2-2', '庄2-20', '庄2-23', '庄2-25', '庄2-26',
#         '庄2-29', '庄2-3', '庄2-32', '庄2-34', '庄2-35',
#         '庄2-36', '庄2-39', '庄2-40', '庄2-43', '庄2-45',
#         '庄2-46', '庄2-47', '庄2-49', '庄2-5', '庄2-52',
#         '庄2-57', '庄2-59', '庄2-61', '庄2-62', '庄2-63',
#         '庄2-64']

pots = ['庄2-35']

#pots = ['庄2-19','庄2']

feature_list = ['年月', '日产油量', '日产水量', '泵效', '地层系数', '泵深']
#feature_list = ['年月', '日产油量', '日产水量', '累积产水量', '月产油量', '月产水量']

# HeatMap_feature_list = ['Top of producing interval', 'Bottom of producing interval',
#                         'Reservoir thickness', 'Pump depth', 'Pump diameter or nozzle',
#                         'Stroke', 'Stroke times', 'Production days',
#                         'Daily oil production', 'Daily water production',
#                         'Daily gas production','Pump efficiency',
#                         'Stratigraphic coefficient', 'Times','Monthly oil production',
#                         'Monthly water production','Monthly gas production',
#                         'Cumulative oil production', 'Cumulative water production',
#                         'Cumulative gas production','Dynamic fluid level',
#                         'Water cut'] #把热力图目标写到最后一个

#PSO参数范围
lb = [0.1, 0.001, 0.0001]
ub = [10, 10, 0.03]

#写入的文件的路径
ResultPath = "WStest.xlsx"



