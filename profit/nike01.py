# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
from selenium import webdriver
from selenium.webdriver import ChromeOptions
import time
import os

# WebDriver
option = ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation'])
wd = webdriver.Chrome('D:/chromedriver.exe', options=option)
wd.implicitly_wait(10)
mainWindow = wd.current_window_handle

# login
url = 'https://m.tb.cn/h.4qoFk9q?sm=3a3f78'
wd.get(url)

wd.switch_to.frame('sufei-dialog-content')  # iframe

wd.find_element_by_xpath("//*[@id='fm-login-id']").send_keys('13679137060')
password = wd.find_element_by_xpath("//*[@id='fm-login-password']").send_keys('zheng262404')

wd.find_elements_by_class_name('fm-button fm-submit password-login').click()
wd.switch_to.default_content()

pass



