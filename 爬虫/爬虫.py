# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import os
import urllib

fp = open('url.txt', 'w', encoding='utf-8')
name1 = os.listdir('G:/GLADS/1948~1978/1')
name2 = os.listdir('G:/GLADS/1979~2014/1')
name = name1 + name2
for i in range(len(name)):
    name[i] = name[i][:33]

url = []
for i in range(len(name)):
     url.append('https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_CLSM025_D.2.0/{}/{}/{}'.format(name[i][17:21], name[i][21:23], name[i]))
     # 'https://hydro1.gesdisc.eosdis.nasa.gov/data/GLDAS/GLDAS_CLSM025_D.2.0/1970/05/GLDAS_CLSM025_D.A19700510.020.nc4'

fp.write("\n".join(url))
fp.close()
