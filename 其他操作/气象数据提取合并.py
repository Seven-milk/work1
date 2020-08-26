# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 将站点数据进行合并
# 站点 53740, 53754, 53646, 53735, 日平均气温, 日降水
import pandas as pd


def data_write(x, filename):
    if type(x) is pd.core.frame.DataFrame:
        writer = pd.ExcelWriter(filename)
        x.to_excel(writer, float_format='%.3f')
        writer.save()
        writer.close()
    else:
        test_data = pd.DataFrame(x)
        writer = pd.ExcelWriter(filename)
        test_data.to_excel(writer, float_format='%.3f')
        writer.save()
        writer.close()


years = list(range(1970, 2014))
# result = pd.DataFrame(columns=('年','月','日','平均气压','日最高气温','日最低气温','平均相对湿度','平均风速','日照时数'))
numbers = [53740, 53754, 53646, 53735]
for j in range(len(numbers)):
    result = pd.DataFrame(columns=('区站号', '年', '月', '日', '平均气温', '日降水'))
    for i in range(len(years)):
        file_name = 'F:/工作/赵香桂/'+str(years[i])+'年.xlsx'
        data1 = pd.read_excel(file_name, sheet_name='温度', usecols=[0, 4, 5, 6, 7])
        data2 = pd.read_excel(file_name, sheet_name='降水', usecols=[0, 9], names=["区站号", "日降水"], header=None)
        read1 = data1[data1['区站号'] == numbers[j]]
        read2 = data2[data2['区站号'] == numbers[j]]
        read = pd.concat([read1, read2["日降水"]], axis=1)
        result = result.append(read)
    data_write(result, '{}.xlsx'.format(numbers[j]))
