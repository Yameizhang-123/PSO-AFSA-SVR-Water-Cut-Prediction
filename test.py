import os

import pandas as pd

from openpyxl import load_workbook
excel_name = "test.xlsx"

for i in range (0,4):

    writer = pd.ExcelWriter(excel_name,engine='openpyxl')# pylint: disable=abstract-class-instantiated

    one_img_info = ['file', 1]
    one_img_info = pd.DataFrame(one_img_info)
    # 创建新excel
    if not os.path.exists(excel_name):

        one_img_info.to_excel(writer, sheet_name=i, index = False)

    else:
        # 1.向已有excel创建新sheet
        book = load_workbook(excel_name)
        writer.book = book
        one_img_info.to_excel(writer, sheet_name=i, index = False)
        # 2.向已有excel添加新行(注释上面三行)
        # df= pd.DataFrame(pd.read_excel(excel_name,sheet_name=dir_i), index=None)
        # out_df = pd.concat([df,one_img_info],axis=0)
        # out_df.to_excel(writer, sheet_name=dir_i, index = False)
    writer.save()
    writer.close()
