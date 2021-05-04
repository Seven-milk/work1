# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import time
import os
import random

# download set
download_path = "G:/NDVI"  # set download path here
pagenumber = 116
print(f"page enumber: {pagenumber}")

# WebDriver
option = ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation'])

# proxy
proxy_arr = [
     '--proxy-server=http://220.181.111.37:80',
     # '--proxy-server=http://49.70.99.163:9999',
     # '--proxy-server=http://160.19.232.85:3128',
     # '--proxy-server=http://183.166.123.112:9999',
     # '--proxy-server=http://117.94.140.3:9999',
     # '--proxy-server=http://114.233.169.40:8073',
     # '--proxy-server=http://124.94.254.134:9999',
     # '--proxy-server=http://113.194.131.151:9999',
     # '--proxy-server=36.248.132.196:9999',
     # '--proxy-server=http://125.46.0.62:53281',
     # '--proxy-server=http://219.239.142.253:3128',
     # '--proxy-server=http://119.57.156.90:53281',
     # '--proxy-server=http://60.205.132.71:80',
     # '--proxy-server=https://139.217.110.76:3128',
     # '--proxy-server=https://116.196.85.150:3128'
 ]
# proxy = random.choice(proxy_arr)
# print(proxy)
# option.add_argument(proxy)

wd = webdriver.Chrome('G:/chromedriver.exe', options=option)
wd.implicitly_wait(10)
mainWindow = wd.current_window_handle

# login
wd.get('http://www.gscloud.cn/sources/accessdata/344?pid=333')
wd.find_element_by_xpath("//*[@class='user-name user-name-login']").click()

email = wd.find_element_by_id("email")
password = wd.find_element_by_id("password")

email.send_keys('786909151@qq.com')
password.send_keys('Zheng262404123')

captcha = wd.find_element_by_xpath('//*[@id="id_captcha_1"]')
captcha_sj = input('input verification code:').strip()  # then set download path here
captcha.send_keys(captcha_sj)

dr_buttoon = wd.find_element_by_id("btn-login").click()

# back
time.sleep(3)
wd.back()
time.sleep(3)
wd.back()
time.sleep(5)

# delete files_downloading before
downloading_before = [os.remove(os.path.join(download_path, file)) for file in os.listdir(download_path)
                      if file[-11:] == '.crdownload']

# skip set
skip = 0
skip_i = 0

# download
for i in range(pagenumber - 1):
    # page file
    page_files_name = [e.text for e in wd.find_elements_by_xpath\
        ("//tr[starts-with(@class, 'dlv-row')]/td[2]//div[@style='text-align: center;']")]
    page_download_buttons = wd.find_elements_by_xpath\
        ("//tr[starts-with(@class, 'dlv-row')]/td[2]/following-sibling::*/div/div/p[2]/img")
    paging_button = wd.find_element_by_xpath("//*[@class='l-btn-empty pagination-next']")

    # skip
    if skip > i + 1:
        # paging
        _ = os.system("cls")
        print(f"skip page{skip_i + 1}")
        paging_button.click()
        time.sleep(5)
        skip_i += 1
        continue

    # check the file have not been download
    while True:
        downloaded_file = set([file[:-4] for file in os.listdir(download_path) if file[-4:] == '.TIF'])
        downloading_file = set([file[:-15] for file in os.listdir(download_path) if file[-11:] == '.crdownload'])
        nobutton_index = []
        for index in range(len(page_files_name)):
            page_file_name = page_files_name[index]
            if not (page_file_name in downloaded_file or page_file_name in downloading_file):
                nobutton_index.append(index)

        # print start downloads
        _ = os.system("cls")
        print(f"All page enumber: {pagenumber}")
        print(f'page{i + 1} is downloading: which contains {len(page_download_buttons)} files, and'
              f' {len(nobutton_index)} files have not been click, {len(downloading_file)} is downloading')
        print(f'{int((1 - (len(downloading_file) + len(nobutton_index)) / len(page_files_name)) * 100)} %...')

        # download nobutton file
        if len(nobutton_index) != 0:
            for nobutton in nobutton_index:
                page_download_buttons[nobutton].click()
                time.sleep(6)  # wait for response
                handles = wd.window_handles
                if len(handles) > 1:
                    for j in range(1, len(handles)):
                        wd.switch_to.window(handles[1])
                        wd.close()
                        wd.switch_to.window(mainWindow)

        # if this page is downloaded
        if set(page_files_name) <= downloaded_file:
            # paging
            print(f'page{i + 1} is downloaded')
            paging_button.click()
            time.sleep(5)
            break
        else:
            # wait 10 sec for download
            time.sleep(60)

# TODO 删除()，即重复内容
# TODO 增加skip便利的方式：找到页数跳转的地方，send_keys，然后点击，然后for (skip, pagenumber - 1)
