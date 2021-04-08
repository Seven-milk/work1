# code: utf-8
# author: "Xudong Zheng"
# email: Z786909151@163.com
# download GLDAS by cookie request method
import os
import re
import requests
from multiprocessing import Pool

# general set
# root = "E:/"
# home = os.path.join(root, "GLDAS_Noah_3hourly")
# home = "G:/GLDAS_NOAH"
home = "F:/data/GLDAS/GLDAS Noah Land Surface Model L4 3 hourly 0.25 x 0.25 degree V2.1 (GLDAS_NOAH025_3H)"
URL = os.path.join(home, "subset_GLDAS_NOAH025_3H_2.1_20210405_032035.txt")

cookie = {
    "_ce.s": "v11.rlc~1616927269668~v11ls~false~v~061d4664fd91e177c5ccaa33472db193789e60f1~vv~7~ir~1",
    "_ga": "GA1.4.698647544.1616905533",
    "_gid": "GA1.4.1749749673.1617592003",
    "_urs-gui_session": "19b9e61a886bb83dc32f647aa604b87f",
    "urs_guid_ops": "65cb2506-b9f2-476f-8c8a-2707adcaef94",
    "urs_user_already_logged": "yes"
}

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
# start = input("input the start date, such as 19480101.0300")
start = ""  # 19480425.1500

if start == "":
    index = 0
    print("start: ", re.search(r"\d{8}\.\d{4}", file_name[0])[0])
else:
    index = [start in name for name in file_name].index(True)
    print("start: ", start)
urls = urls[index:]
file_name = file_name[index:]


# download function
def download(url, filename):
    '''
    input:
        url: list
        filename: list
    '''
    print(f"start download {filename}")
    try:
        response = requests.get(url, cookies=cookie)
        f = open(filename, 'wb')
        f.write(response.content)
        f.close()
        print('contents of URL written to ' + filename)

    except:
        # append write fail urls in fail_url.txt
        print('Error to connect' + filename)
        with open(os.path.join(home, "fail_url.txt"), 'a') as f:
            f.write(url + "\n")


# download pdf file
# download(url_pdf, pdf_name)


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
    # mpdownload()
    serial_download()
