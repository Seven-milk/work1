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

# check date: check if the date is consecutive
# 1) change the date into pd.TimeIndex
# 2) diff = date[1:] - date[:-1]
# 3) compare diff with input time_interval
# 4) if diff != time_interval, the date is not consecutive with its prior date

import os
import re
from netCDF4 import Dataset
import Workflow
import time
import pandas as pd
import numpy as np


class CheckCrawler(Workflow.WorkBase):
    ''' Work, check whether the crawler have downloaded all files and if all files can be open, namely downloaded
    compelely

    URL - all files that should be downloaded
    home - files that have been downloaded
    URL - home = Undownloaded file
    data stracture: set(disorder) and list(order)

    '''

    def __init__(self, home, URL, r, time_format='%Y%m%d.%H%S'):
        ''' init function
        input:
            home: home path, where downloaded files exist
            URL: url path, a text file includes all urls
            r: <class 're.Pattern'>, regular experssion to identify unique file/url, use re.compile(r"...") to build it
                e.g. 19980101.0300 - r = re.compile(r"\d{8}\.\d{4}")
                e.g. 19980101 - r = re.compile(r"\d{8}")
            time_format: str, change str to time format, match the r
                e.g. 19980101 - time_format = '%Y%m%d'
                e.g. 19980101.0300 - time_format = '%Y%m%d.%H%S'

        note:
        1) this function focus on .nc4 file, you can change it in file_name
        2) r should match the filenames which have been downloaded, if not, modify it in regular_expression_url()
        3) urls in URL file should have the same format, if not, urls with different format should be in the beginning
           of the file

    ouput:
        url_notdownload.txt: text file, including urls which not be downloaded
        files_not_opened.txt: text file, including files which can not be opened
        dates_not_consecutive.txt: text file, including date which not consecutive

        '''
        self.home = home
        self.URL = URL
        self.r = r
        self.time_format = time_format

    def __call__(self):
        ''' Implement WorkBase.__call__ '''
        # checkDownload
        print("start check if all files in urls has been Download")
        ret_download = self.checkDownload()

        if ret_download == True:
            # checkFilesComplete
            print("start check if all downloaded files can be opened")
            self.checkFilesComplete()

            # checkDate
            print("start check if all files is consecutive in date")
            self.checkDate()

            print("check complete")

        else:
            print("check complete")

    def checkDownload(self):
        ''' check whether the crawler have downloaded all files '''
        # general set
        home = self.home
        URL = self.URL
        r = self.r

        # open URL file and read it to a list
        with open(URL, 'r') as file:
            urls = file.read()
            urls = urls.split("\n")

        # file_name_all
        file_name_all, skip = self.regular_expression_url(urls, r)

        # file_name_downloaded
        file_name_downloaded = [r.search(path)[0] for path in os.listdir(home) if path[-4:] == ".nc4"]

        # change list to set
        file_name_all = set(file_name_all)
        file_name_downloaded = set(file_name_downloaded)

        # difference between the two sets
        file_name_notdownloaded = file_name_all - file_name_downloaded
        file_name_notdownloaded = list(file_name_notdownloaded)

        # Numbers of files not downlaoded
        if len(file_name_notdownloaded) == 0:
            print("All urls have been downloaded!")
            return True
        else:
            print("There are " + f"{len(file_name_notdownloaded)}" + " files not downloaded!")

            # make sure the order is right
            file_name_all, _ = self.regular_expression_url(urls, r, p=False)

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
                print("url_notdownload have saved to url_notdownload.txt")

            return False

    def regular_expression_url(self, urls, r, p=True):
        ''' search filename in urls based on r, it should be consistent with filename in downloaded files, exclude urls
        that does not consistent with r(such as, it could contains README.PDF in download urls)

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
                file_name_all.append(r.search(url)[0])
            except:
                skip += 1
                if p == True:
                    print(f"Url(in urls file) doesn't match regular experssion!: {url}")

        return file_name_all, skip

    def checkFilesComplete(self):
        ''' open every file to check if it is downloaded completely '''
        # general set
        home = self.home
        r = self.r

        # set files based on start
        start = input("input start time that contains in nc file name, e.g. 19980101.0300, if start from begin,"
                      "input enter\n:")
        files = [file for file in os.listdir(home) if file.endswith('.nc4')]
        files.sort(key=lambda x: float(r.search(x)[0]))  # sort files
        index_start = 0 if start == "" else [start in name for name in files].index(True)
        files = files[index_start:]

        # ask if save file is exist, because the save mode == 'a'
        while True:
            if os.path.exists(os.path.join(home, 'files_not_opened.txt')):
                rm = input("files_not_opened.txt is exist in home path, does rm it? True or False:")
                if rm == True:
                    os.remove(os.path.join(home, 'files_not_opened.txt'))
                    break
                else:
                    time.sleep(3)
            else:
                break

        # try to open all files
        for i in range(len(files)):
            try:
                f = Dataset(os.path.join(home, files[i]), 'r')
                f.close()
                print('can open: ',  files[i])
            except:
                print(f'File error: {files[i]} can not open!')
                # save
                with open(os.path.join(home, 'files_not_opened.txt'), 'a', encoding='utf-8') as f:
                    f.write(files[i])

        print("files can not be opened have saved to files_not_opened.txt")

    def checkDate(self):
        ''' check if all files is consecutive in date '''
        # general set
        home = self.home
        r = self.r
        time_interval = input("input time_interval: such as 3H, 1D:")

        # list for date and change it to pd.TimeIndex
        files = [file for file in os.listdir(home) if file.endswith('.nc4')]
        files.sort(key=lambda x: float(r.search(x)[0]))  # sort files
        date_files = [self.r.search(file)[0] for file in files]
        date_files = pd.to_datetime(date_files, format=self.time_format)

        # Make difference, begin from the second if difference is equal to time_interval, it is consecutive, if not, the
        # date is not consecutive with the prior date
        diff = date_files[1:] - date_files[:-1]
        time_interval = pd.Timedelta(time_interval)
        ret = (diff == time_interval)
        false_num = len(ret) - sum(ret)
        false_index = np.argwhere(ret == False).flatten()
        false_date = date_files[false_index + 1]
        if false_num == 0:
            print(f"The date is consecutive, start from {date_files[0]}, end with {date_files[-1]}")
        else:
            print(f"There are {false_num} dates is not consecutive, the dates is presented below, and the date have"
                  f" saved to dates_not_consecutive.txt")
            print("note: check begin from the second date, the dates list here are not consecutive with its prior date")
            print("\n".join([str(i) for i in false_date]))
            with open(os.path.join(home, 'dates_not_consecutive.txt'), 'w', encoding='utf-8') as f:
                f.write("note: check begin from the second date, the date in the file is not consecutive with the prior"
                        " date")
                f.write("\n".join([str(i) for i in false_date]))


if __name__ == '__main__':
    # general set
    home = "E:/GLDAS_NOAH"
    URL = os.path.join(home, "subset_GLDAS_NOAH025_3H_2.0_20210328_114227.txt")
    # r = re.compile(r"LABEL.*\d\.nc4")
    r = re.compile(r"\d{8}\.\d{4}")
    cc = CheckCrawler(home, URL, r, time_format='%Y%m%d.%H%S')
    cc()
