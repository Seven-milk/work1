# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

# check whether the file is downloaded completely:
# 1) url file(URL) -> file name(match the download files'name using regular expression) ->
# 2) set1, download file(file in home) ->
# 3) file name -> set2, set1 - set2 ->
# 4) file name not download ->
# 5) search index in url file -> url not download (url notdownload.txt)
# check whether the file can be opened
# 1) try to open every file from "start"
# 2) if it can be opened, pass it
# 3) if it can not be opened, save the file name to the result

import os
import re
from netCDF4 import Dataset


def check_crawler(home, URL, r):
    ''' check whether the crawler have downloaded all files
    URL - all files that should be downloaded
    home - files that have been downloaded
    URL - home = Undownloaded file
    data stracture: set(disorder) and list(order)

    input:
        home: home path, where downloaded files exist
        URL: url path, a text file includes all urls
        r: <class 're.Pattern'>, regular experssion to identify unique file/url, use re.compile(r"...") to build it

        note:
        1) this function focus on .nc4 file, you can change it in file_name
        2) r should match the filenames which have been downloaded, if not, modify it in regular_expression_url()
        3) urls in URL file should have the same format, if not, urls with different format should be in the beginning
           of the file

    ouput:
        url_notdownload: text file, including urls which not be downloaded

    '''

    # file_name_all in URL using r
    def regular_expression_url(urls, r, p=True):
        ''' search filename in urls based on r, it should be consistent with filename in downloaded files
        input:
            urls/r: same as input of check_crawler
            p: whether print warning

        output:
            file_name_all: list, urls -> file_name with same format as r
        '''
        file_name_all = []
        skip = 0
        for url in urls:
            try:
                file_name_all.append(r.search(url)[0][6:])
            except:
                skip += 1
                if p == True:
                    print(f"Url(in urls file) doesn't match regular experssion!: {url}")

        return file_name_all, skip

    # general set
    home = home
    URL = URL

    # open URL file and read it to a list
    with open(URL, 'r') as file:
        urls = file.read()
        urls = urls.split("\n")

    # file_name_all
    file_name_all, skip = regular_expression_url(urls, r)

    # file_name_downloaded
    file_name_downloaded = [path for path in os.listdir(home) if path[-4:] == ".nc4"]

    # change list to set
    file_name_all = set(file_name_all)
    file_name_downloaded = set(file_name_downloaded)

    # difference set
    file_name_notdownloaded = file_name_all - file_name_downloaded
    file_name_notdownloaded = list(file_name_notdownloaded)

    # Numbers of files not downlaoded
    if len(file_name_notdownloaded) == 0:
        print("there is no url which have not been downloaded!")
        return None
    else:
        print("There are " + f"{len(file_name_notdownloaded)}" + " files not downloaded!")

    # make sure the order is right
    file_name_all, _ = regular_expression_url(urls, r, p=False)

    # search url_notdownload
    print("searching...")
    url_notdownload = []

    for i in range(len(file_name_notdownloaded)):
        index_ = [file_name_notdownloaded[i] in name for name in file_name_all].index(True)
        index_ += skip
        print(f"{int(i / len(file_name_notdownloaded) * 100)}%...")
        url_notdownload.append(urls[index_])

    print("completed!")

    # join url into a str
    url_notdownload = "\n".join(url_notdownload)

    # save
    with open(os.path.join(home, 'url_notdownload.txt'), 'w', encoding='utf-8') as f:
        f.write(url_notdownload)


def check_downloaded_completely(home, start):
    ''' open every file to check if it is downloaded completely '''
    files = [file for file in os.listdir(home) if file.endswith('.nc4')]
    index_start = 0 if start == "" else [start in name for name in files].index(True)

    files = files[index_start:]

    for i in range(len(files)):
        try:
            f = Dataset(os.path.join(home, files[i]), 'r')
            f.close()
            print('can open: ',  files[i])
        except:
            raise SyntaxError(f'File error: {files[i]} can not open!')


if __name__ == '__main__':
    # general set
    home = "D:/GLDAS_NOAH"
    URL = os.path.join(home, "subset_GLDAS_NOAH025_3H_2.0_20210328_114227.txt")
    # r = re.compile(r"LABEL.*\d\.nc4")
    r = re.compile(r"\d{8}\.\d{4}")
    # check_crawler(home, URL, r)
    check_downloaded_completely(home, start='')