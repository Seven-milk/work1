# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import numpy as np
import openpyxl, xlrd
import pandas as pd
# 河南省，广州、深圳、杭州、温州市
def search(sheet_name, tel_name, name="姓名"):
    data = pd.read_excel('F:/文件/zt/2020-2-20/西安理工大学“两省四市”学生信息台账.xlsx', sheet_name)
    country = data["籍贯"]
    result = pd.DataFrame(np.zeros((data.shape[0],5)), columns=["姓名","性别","班级","籍贯","学生联系方式"])
    for i in range(len(country)):
        if (country[i].find("河南"or"广州"or"深圳"or"杭州"or"温州")!=-1):
            result.loc[i,"姓名"] = data.loc[i,name]
            result.loc[i,"性别"] = data.loc[i,"性别"]
            result.loc[i,"班级"] = data.loc[i,"班级"]
            result.loc[i,"籍贯"] = data.loc[i,"籍贯"]
            result.loc[i,"学生联系方式"] = data.loc[i,tel_name]
    result = result[result["姓名"]!=0]
    return result

result1 = search(sheet_name="电气电网", tel_name="个人电话")
result2 = search(sheet_name="191", tel_name="本人电话",name=" 姓名")
result3 = search(sheet_name="192", tel_name="本人电话")
result4 = search(sheet_name="193", tel_name="本人电话")

writer = pd.ExcelWriter('zt.xlsx')
result4.to_excel(writer)
writer.save()
writer.close()
