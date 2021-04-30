# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import time
import os

# download set
download_path = "G:/NDVI"  # set download path here
pagenumber = 116
print(f"page enumber: {pagenumber}")

# WebDriver
option = ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation'])
wd = webdriver.Chrome('D:/chromedriver.exe', options=option)
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

# download
for i in range(pagenumber - 1):
    # page file
    page_files_name = [e.text for e in wd.find_elements_by_xpath\
        ("//tr[starts-with(@class, 'dlv-row')]/td[2]//div[@style='text-align: center;']")]
    page_download_buttons = wd.find_elements_by_xpath\
        ("//tr[starts-with(@class, 'dlv-row')]/td[2]/following-sibling::*/div/div/p[2]/img")
    paging_button = wd.find_element_by_xpath("//*[@class='l-btn-empty pagination-next']")

    # check the file have not been download
    while True:
        downloaded_file = set([file[:-4] for file in os.listdir(download_path) if file[-4:] == '.TIF'])
        downloading_file = set([file[:-11] for file in os.listdir(download_path) if file[-11:] == '.crdownload'])
        nobutton_index = []
        i = 0
        for page_file_name in page_files_name:
            if not (page_file_name in downloaded_file or page_file_name in downloading_file):
                nobutton_index.append(i)
                i += 1

        # print start downloads
        _ = os.system("cls")
        print(f"All page enumber: {pagenumber}")
        print(f'page{i + 1} is downloading: which contains {len(page_download_buttons)} files, and'
              f'{len(nobutton_index)} files have not been downloaded')
        print(f'{int((1 - (len(downloading_file) + len(nobutton_index)) / len(page_files_name)) * 100)} %...')

        # download nobutton file
        if len(nobutton_index) != 0:
            for nobutton in nobutton_index:
                page_download_buttons[nobutton].click()
                time.sleep(5)  # wait for response
                handles = wd.window_handles
                if len(handles) > 1:
                    for i in range(1, len(handles)):
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
            time.sleep(10)
