# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# 爬取给定网站的zt要的串讲音频和文本
import requests

# 爬取音频
url = [f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj0{i}.mp3' for i in range(1, 10)]
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj{i}.mp3' for i in range(10, 25)])
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncjshengsipin0{i}.mp3' for i in range(1, 10)])
url.extend(['http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj25.mp3'])
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj--dyp{i}.mp3' for i in range(18, 20)])
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj-dyp{i}.mp3' for i in range(20, 53)])
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj-egp0{i}.mp3' for i in range(1, 6)])
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj-egp0{i}.mp3' for i in range(7, 10)])
url.extend(['http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj-egp06-t.mp3'])
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj-egp{i}.mp3' for i in range(1, 15)])
url.extend([f'http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj-csp0{i}.mp3' for i in range(1, 5)])
print(url)
for url_str in url:
    r = requests.get(url_str)
    with open(url_str.split('/')[-1], 'wb') as mp4:
        mp4.write(r.content)
        print(url_str.split('/')[-1] + ' is already download')

# 爬取文本 http://files.xianmifw.com/kanbufy/fjbd_zhengfanianchujing/zfncj01.doc
for url_str in url:
    url_str1 = url_str.replace('mp3', 'doc')
    r = requests.get(url_str1)
    with open(url_str1.split('/')[-1], 'wb') as doc:
        doc.write(r.content)
        print(url_str1.split('/')[-1] + ' is already download')
