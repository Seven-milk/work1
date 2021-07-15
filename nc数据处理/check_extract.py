# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

# check values: check if the extract variable(given region by coord) from .nc4 file is right
# 1) select {random_num[= 100]} files/time and points
# 2) cal index - files/time(row), points(col)
# 3) extract_data[row, col], note the first col is time, col select from 1 to len - 1
# 4) corresponding nc file - selected based on time(in filename) and coord(cal index from self.cal_index)
# 5) compare - check_result = (extract_data == corresponding nc file)
# do this job for {check_num [= 1]} times ti make sure extract right
# theory: based on two extract method(extract function, such as extract_nc_wwr_mp) and [time/filenames, coord] within
# random files/points, two extract file can be compared for verifying

# check data: check if the extract variable date is same with .nc4 files date(from file name)
# 1) change the two date into pd.TimeIndex
# 2) compare len
# 3) compare value -> time_extract_data[i] == time_source_data[i]
# 4) count True and return accuracy

import numpy as np
import os
from netCDF4 import Dataset
import Workflow
import time
import re
import pandas as pd
import useful_func


class CheckExtract(Workflow.WorkBase):
    ''' Work, Check if the Extract file (.npy) is consistent with source file (.nc4) '''

    def __init__(self, extract_data_path, source_data_path, coord_path, variable_name, r, precision=3, check_num=1,
                 time_format="%Y%m%d.%H%S"):
        ''' init function
        input:
            extract_data_path: .npy file, the extract file from nc files based on before extract processing
            source_data_path: home path for source nc file
            coord_path: .txt, coord file path
            variable_name: corresponding to extract_data, the variable which is extracted into extract file
            r: <class 're.Pattern'>, regular experssion to identify time, use re.compile(r"...") to build it
                e.g. 19980101 - r = re.compile(r"\d{8}")
                e.g. 19980101.0300 - r = re.compile(r"\d{8}\.\d{4}")
            time_format: str, change str to time format, match the r
                e.g. 19980101 - time_format = '%Y%m%d'
                e.g. 19980101.0300 - time_format = '%Y%m%d.%H%S'
            precision: int, the minimum precision of lat/lon, to match the lat/lon of source nc file
            check_num: check times

        output:
            print the check result, if accuracy is not equal to 100%(false != 0), attention the extract processing!
        '''
        self.extract_data_path = extract_data_path
        self.source_data_path = source_data_path
        self.coord_path = coord_path
        self.variable_name = variable_name
        self.r = r
        self.check_num = check_num
        self.precision = precision
        self.time_format = time_format

        # read file
        self.extract_data = np.load(self.extract_data_path, mmap_mode='r')
        self.coord = pd.read_csv(self.coord_path, sep=",")  # read coord(extract by fishnet)
        self.coord = self.coord.round(self.precision)  # coord precision correlating with .nc file lat/lon
        self.nc_path = [self.source_data_path + "/" + d for d in os.listdir(self.source_data_path) if d[-4:] == ".nc4"]

    def __call__(self):
        ''' Implement WorkBase.__call__ '''
        # check date
        print("-----------Check date-----------")
        ret_check_date = self.checkDate()
        if isinstance(ret_check_date, list):
            print('extract number is equal to nc number')
            print('-----------------------------------')
            print('false date = ', ret_check_date[1])
            print('-----------------------------------')
            print(f'date_accuracy = {ret_check_date[2]}%')
        else:
            if ret_check_date > 0:
                print("extract data is more than nc number")
            elif ret_check_date < 0:
                print("extract data is less than nc number")

        print('-----------------------------------')

        # check value
        print("-----------Check value-----------")
        for i in range(self.check_num):
            check_result, false_num, accuracy = self.checkValues()
            # print
            print(f'check{i}: false_num = {false_num}, accuracy = {accuracy}%')

        print('-----------------------------------')

    def checkValues(self):
        ''' check values '''
        # set random index: 100 random files/time, 100 random points
        random_num = 100
        index_random_file = np.random.randint(0, self.extract_data.shape[0], random_num)  # row - files/time
        index_random_point = np.random.randint(1, self.extract_data.shape[1] - 1, random_num)  # col - points

        time_random_file = [self.r.search("%.8f" % self.extract_data[i, 0])[0] for i in index_random_file]
        random_coord = self.coord.iloc[index_random_point - 1, :]
        random_nc_path = [nc_path for time in time_random_file for nc_path in self.nc_path if time in nc_path]

        # random extract_data
        random_extract_data = useful_func.extractIndexArray(index_random_file, index_random_point, self.extract_data)

        # corresponding source data
        random_source_data = np.zeros_like(random_extract_data)
        for i in range(len(index_random_file)):
            lat_index, lon_index = self.cal_index(random_nc_path[i], random_coord)
            f = Dataset(random_nc_path[i], 'r')
            for j in range(len(random_coord)):
                random_source_data[i, j] = f.variables[self.variable_name][0, lat_index[j], lon_index[j]]
            f.close()

        # compare - check.result
        check_result = (random_extract_data == random_source_data)
        false_num = check_result.size - check_result.sum()
        accuracy = (1 - false_num / random_num / random_num) * 100

        return check_result, false_num, accuracy

    def checkDate(self):
        ''' check whether the extract date is same as nc file '''
        # extract time from extract data and source data(fname)
        time_extract_data = [self.r.search("%.8f" % self.extract_data[i, 0])[0] for i in range(len(self.extract_data))]
        time_source_data = [self.r.search(nc_path)[0] for nc_path in self.nc_path]

        # sort
        time_source_data.sort(key=lambda x: float(x))
        time_extract_data.sort(key=lambda x: float(x))

        # same num
        ret_same_num = len(time_extract_data) - len(time_source_data)

        # to date
        if ret_same_num == 0:
            time_extract_data = pd.to_datetime(time_extract_data, format=self.time_format)
            time_source_data = pd.to_datetime(time_source_data, format=self.time_format)
            ret_same_date = [time_extract_data[i] == time_source_data[i] for i in range(len(time_extract_data))]
            false_num = len(ret_same_date) - sum(ret_same_date)
            false_date = []

            for i in range(len(ret_same_date)):
                if ret_same_date[i] == False:
                    false_date.append({'extract': time_extract_data[i], 'source': time_source_data[i]})

            ret_same_date_accuracy = (1 - false_num / len(ret_same_date)) * 100
            return [ret_same_num, false_date, ret_same_date_accuracy]
        else:
            return ret_same_num

    def cal_index(self, ncfile: str, coord: pd.DataFrame):
        ''' calculate the index of lat/lon in coord from source nc file
        input:
            ncfile: str, the path of source ncfile to cal index
            coord: pd.Dataframe, coord(extract by fishnet)

        output:
            lat/lon_index: lat/lon index of coord in source nc file
        '''
        f = Dataset(ncfile, 'r')
        Dataset.set_auto_mask(f, False)
        lat_index = []
        lon_index = []
        lat = f.variables["lat"][:]
        lon = f.variables["lon"][:]
        for j in range(len(coord)):
            lat_index.append(np.where(lat == coord["lat"].iloc[j])[0][0])
            lon_index.append(np.where(lon == coord["lon"].iloc[j])[0][0])
        f.close()
        return lat_index, lon_index


if __name__ == '__main__':
    extract_data_path = 'H:/research/flash_drough/GLDAS_Noah/CanopInt_inst_19480101.0300_20141231.2100.npy'
    source_data_path = 'E:/GLDAS_NOAH'
    coord_path = "H:/GIS/Flash_drought/coord.txt"
    variable_name = 'CanopInt_inst'
    r = re.compile(r"\d{8}\.\d{4}")
    # r = re.compile(r'\d{8}')
    ce = CheckExtract(extract_data_path=extract_data_path, source_data_path=source_data_path, coord_path=coord_path,
                      variable_name=variable_name, r=r, precision=3, check_num=10, time_format="%Y%m%d.%H%S")
    ce()