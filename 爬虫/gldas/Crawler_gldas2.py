# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# Set the URL string to point to a specific data URL. Some generic examples are:
#   https://servername/data/path/file
#   https://servername/opendap/path/file[.format[?subset]]
#   https://servername/daac-bin/OTF/HTTP_services.cgi?KEYWORD=value[&KEYWORD=value]
# URL = 'your_URL_string_goes_here'
#
# # Set the FILENAME string to the data file name, the LABEL keyword value, or any customized name.
# FILENAME = 'your_filename_string_goes_here'
#
#
#
# result = requests.get(URL)
# try:
#     result.raise_for_status()
#     f = open(FILENAME, 'wb')
#     f.write(result.content)
#     f.close()
#     print('contents of URL written to ' + FILENAME)
# except:
#     print('requests.get() returned an error code ' + str(result.status_code))

import requests
import os
import re

# general set
root = "E:/"
home = os.path.join(root, "GLDAS_Noah_3hourly")
URL = os.path.join(home, "subset_GLDAS_NOAH025_3H_2.0_20210328_114227.txt")

# open url file and read url in urls
with open(URL, 'r') as file:
    urls = file.read()
    urls = urls.split("\n")

# distinguish url of pdf and nc4 file
url_pdf = urls[0]
urls = urls[1:]
pdf_name = 'README_GLDAS2.pdf'
pdf_name = os.path.join(home, pdf_name)
file_name = [re.search(r"LABEL.*\d\.nc4", url)[0][6:] for url in urls]
file_name = [os.path.join(home, file) for file in file_name]

# input start date and find the index in urls/file_name
start = input("input the start date, such as 19480101.0300")
if start == "":
    index = 0
    print("start: ", re.search(r"\d{8}\.\d{4}", file_name[0])[0])
else:
    index = [start in name for name in file_name].index(True)
    print("start: ", start)


# download function
def download(url, filename):
    print(f"start download {filename}")
    result = requests.get(url)
    try:
        result.raise_for_status()
        f = open(filename, 'wb')
        f.write(result.content)
        f.close()
        print('contents of URL written to ' + filename)
    except:
        print('requests.get() returned an error code ' + str(result.status_code))


# download nc file
for i in range(index, len(file_name)):
    download(urls[i], file_name[i])


# download pdf file
# download(url_pdf, pdf_name)



