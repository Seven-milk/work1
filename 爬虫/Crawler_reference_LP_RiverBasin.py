# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 未完成，主要进行了网页结构分析，网页解析还没完成
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import re

home = "C:/Users/wlbless/Desktop/黄土高原"
# merge two txt file to a excel
# path1 = os.path.join(home, "savedrecs.txt")
# path2 = os.path.join(home, "savedrecs2.txt")
#
# df1 = pd.read_table(path1)
# df2 = pd.read_table(path2)
#
# df = df1.append(df2)
# df.to_excel(f"{home}/savedrecs_append.xlsx")
# search info
path = f"{home}/savedrecs_append.xlsx"
df = pd.read_excel(path)

# request info
header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:86.0) Gecko/20100101 Firefox/86.0"}
cookie = {
    "_pk_id": "6b4e56d6-fa16-4a55-9af5-dd5bb32a1378.1608280721.6.1615189314.1615186602.",
    "_pk_ref": "[\"\",\"\",1615186602,\"https://www.cnki.net/\"]",
    "_pk_ses": "*",
    "ASP.NET_SessionId": "1hcdw4a42jxcfnwcorjg3oqi",
    "ASPSESSIONIDQABRCABQ": "FMDACIHCJPCGCCHIOOALLBMM",
    "cnkiUserKey": "aa6760f4-5737-dc62-172b-cbcbc8f9c712",
    "CurrSortFieldType": "desc",
    "Ecp_ClientId": "2200924201903012842",
    "Ecp_ClientIp": "202.200.127.88",
    "Ecp_notFirstLogin": "nsQ9cd",
    "Ecp_session": "1",
    "Hm_lvt_6e967eb120601ea41b9d312166416aa6": "1608280759,1608818332,1608819644",
    "KNS_SortType": "",
    "LID": "WEEvREcwSlJHSldSdmVqMDh6a1dqZVBXaDdPUnlZcG5aQlp4Z3pLZ0Y0RT0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!",
    "RsPerPage": "20",
    "SID_crrs": "125133",
    "SID_kcms": "124119",
    "SID_klogin": "125142",
    "SID_kns": "015123127",
    "SID_kns_new": "kns123115",
    "SID_kns8": "123113",
    "SID_krsnew": "125133",
    "SID_kxreader_new": "011122",
    "UM_distinctid": "1747d62ec1526-0ca351471c9247-4c3f247a-1fa400-1747d62ec1869",
    "x-s3-sid": ">/5E2Ci8fkdi`tkE/A1b3t400"
}

# request
results = []
keyword = df["Z1"][0]
url = f'https://kns.cnki.net/kns/brief/brief.aspx?pagename=ASP.brief_default_result_aspx&isinEn=1&dbPrefix=SCDB&dbCatalog=中国学术文献网络出版总库&ConfigFile=SCDBINDEX.xml&research=off&t=1615188527052&keyValue={keyword}&S=1&sorttype='
res = requests.get(url=url, headers=header, cookies=cookie)

# analysis to extract url
# res.content.decode("utf8","ignore").encode("gbk","ignore")
soup = BeautifulSoup(res.content, 'html.parser')
a_download = soup.find_all('a', class_='briefDl_D')
a_name = soup.find_all('a', class_='fz14')

# for i in range(len(df)):
#     keyword = df["TI"][i]
#     try:
#         params = {}
