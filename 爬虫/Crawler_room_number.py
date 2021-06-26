# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import pandas as pd
import re
import requests
from lxml import etree

url_base = 'http://zfyxdj.xa.gov.cn/zfrgdjpt/jggs.aspx?qy=00&yxbh=0000002411&type=2&page='
pages = list(range(2, 74))  # 74
df = pd.DataFrame(columns=['意向登记号', '申请人姓名', '证件类型', '申请人证件号码'])
xpath = '//tr/following-sibling::*'

for page in pages:
    url = url_base + str(page)
    html = requests.get(url).content
    selector = etree.HTML(html)
    trs = selector.xpath(xpath)
    for tr in trs:
        number = tr.xpath('*')[0].text[:8]
        name = tr.xpath('*')[1].text
        id_type = tr.xpath('*')[2].text
        id = tr.xpath('*')[3].text

        # append
        df = df.append(pd.DataFrame({'意向登记号': [number], '申请人姓名': [name], '证件类型': [id_type], '申请人证件号码': [id]}))

    print(f"page {page} is finished")

df.to_excel("room_number.xlsx")
