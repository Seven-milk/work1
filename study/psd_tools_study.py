# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import psd_tools
from psd_tools import PSDImage

psd = PSDImage.open('F:/工作/书稿5.21/出图/图1.psd')

for layer in psd:
    print(layer)
    if layer.is_group():
        for child in layer:
            print(child)

child = psd[0][0]