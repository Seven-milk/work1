# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import time
import os

option = ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation'])
wd = webdriver.Chrome('G:/chromedriver.exe', options=option)
wd.implicitly_wait(10)

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

# download
download_path = "G:/NDVI"
pagenumber = 116
# TODO 找没下的下载
print(f"page enumber: {pagenumber}")
for i in range(pagenumber - 1):
    time.sleep(3)
    downloads = wd.find_elements_by_class_name("download-img")
    print('------------------------------')
    print(f'page{i + 1} is downloading, which contains {len(downloads)} files')

    # check this page whether has been downloaded
    time.sleep(1)
    page_file_name = set([e.text for e in wd.find_elements_by_xpath(
        "//tr[starts-with(@class, 'dlv-row')]/td[2]//div[@style='text-align: center;']")])  # text
    downloaded_file = set([file[:-4] for file in os.listdir(download_path) if file[-4:] == '.TIF'])
    if page_file_name <= downloaded_file:
        # paging
        print(f'page{i + 1} is downloaded')
        wd.find_element_by_xpath("//*[@class='l-btn-empty pagination-next']").click()
        continue

    # download click
    for download in downloads:
        download.click()
        time.sleep(3)

    # waiting for download: check downloadfile
    while True:
        downloaded_file = set([file[:-4] for file in os.listdir(download_path) if file[-4:] == '.TIF'])
        # if downloaded all
        if page_file_name <= downloaded_file:
            break
        else:
            print(f"downloading {int((len(downloaded_file) - i*10) / len(page_file_name) * 100)} %...")
        # wait 10 sec
        time.sleep(10)

    # paging
    print(f'page{i + 1} is downloaded')
    wd.find_element_by_xpath("//*[@class='l-btn-empty pagination-next']").click()
