# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import psd_tools
from psd_tools import PSDImage
import os

filename = os.listdir('G:/安全生产规章制度')
l = []

for i in range(len(filename)):
    psd = PSDImage.open('G:/安全生产规章制度/{}'.format(filename[i]))
    for j in range(len(psd)):
        if isinstance(psd[j], psd_tools.api.layers.TypeLayer):  # 判断是否为子类
            l.append(psd[j].text)
    l.append('\r')

l2 = ''.join(l)
fp = open('G:/安全生产规章制度/文本集合.txt', 'w', encoding='utf-8')
fp.write(l2)
fp.close()
