# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# redownload file based on a url file
import os
import re
import requests
from multiprocessing import Pool

# general set
home = "G:/GLDAS_NOAH"
# URL = os.path.join(home, "url_notdownload.txt")
# URL = os.path.join(home, "fail_url.txt")
URL = os.path.join(home, "url_notdownload2.txt")

# open url file and read url in urls
with open(URL, 'r') as file:
    urls = file.read()
    urls = urls.split("\n")

file_name = [re.search(r"LABEL.*\d\.nc4", url)[0][6:] for url in urls]
file_name = [os.path.join(home, file) for file in file_name]


# download function
def download(url, filename):
    '''
    input:
        url: list
        filename: list
    '''
    print(f"start download {filename}")
    try:
        response = requests.get(url)
        f = open(filename, 'wb')
        f.write(response.content)
        f.close()
        print('contents of URL written to ' + filename)

    except:
        print('Error to connect' + filename)
        # append write fail urls in fail_url.txt
        with open(os.path.join(home, "fail_url.txt"), 'a') as f:
            f.write(url + "\n")


# download nc file
def serial_download():
    for i in range(len(urls)):
        download(urls[i], file_name[i])


# download nc file by multiprocessing
def mp_download():
    po = Pool(4)  # pool
    for i in range(len(urls)):
        po.apply_async(download, (urls[i], file_name[i]))

    po.close()
    po.join()


if __name__ == "__main__":
    mp_download()
    # serial_download()
