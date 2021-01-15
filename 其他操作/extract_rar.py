# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import rarfile
import zipfile
import os
import multiprocessing

home = "F:\文件\简历"
filename = os.path.join(home, "【021】自荐信6个.rar")
# desPath = home

# 根据文件扩展名，使用不同的库
if filename.endswith('.zip'):
    fp = zipfile.ZipFile(filename)
elif filename.endswith('.rar'):
    fp = rarfile.RarFile(filename)

# 无密码
# fp.extractall()
# fp.close()
# print('No password')
# return

# 破解
dict_home = "D:/密码破解/字典/Blasting_dictionary"
dict_path = [os.path.join(dict_home, i) for i in os.listdir(dict_home) if i.endswith(".txt")]

for dict in dict_path:
    try:
        fpPwd = open(dict, encoding="UTF-8")
        for pwd in fpPwd:
            pwd = pwd.rstrip()
            try:
                fp.extractall(pwd=pwd)
                print('Success! ====>'+pwd)
                fp.close()
                break
            except:
                print(pwd+" is not pwd")
                continue
        fpPwd.close()
    except:
        continue