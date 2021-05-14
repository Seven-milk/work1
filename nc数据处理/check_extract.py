# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

# check if the extract variable(given region by coord) from .nc4 file is right
# 1) select {random_num[= 100]} files/time and points
# 2) cal index - files/time(row), points(col)
# 3) extract_data[row, col], note the first col is time, col select from 1 to len - 1
# 4) corresponding nc file - selected based on time(in filename) and coord(cal index from self.cal_index)
# 5) compare - check_result = (extract_data == corresponding nc file)
# do this job for {check_num [= 1]} times ti make sure extract right
# theory: based on two extract method(extract function, such as extract_nc_wwr_mp) and [time/filenames, coord] within
# random files/points, two extract file can be compared for verifying

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

    def __init__(self, extract_data_path, source_data_path, coord_path, variable_name, r, precision=3, check_num=1):
        ''' init function
        input:
            extract_data_path: .npy file, the extract file from nc files based on before extract processing
            source_data_path: home path for source nc file
            coord_path: .txt, coord file path
            variable_name: corresponding to extract_data, the variable which is extracted into extract file
            r: <class 're.Pattern'>, regular experssion to identify time, use re.compile(r"...") to build it
                e.g. 19980101 - r = re.compile(r"\d{8}")
                e.g. 19980101.0300 - r = re.compile(r"\d{8}\.\d{4}")
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

        # read file
        self.extract_data = np.load(self.extract_data_path, mmap_mode='r')
        self.coord = pd.read_csv(self.coord_path, sep=",")  # read coord(extract by fishnet)
        self.coord = self.coord.round(self.precision)  # coord precision correlating with .nc file lat/lon
        self.nc_path = [self.source_data_path + "/" + d for d in os.listdir(self.source_data_path) if d[-4:] == ".nc4"]

    def run(self):
        ''' Implement WorkBase.run() '''
        # check
        for i in range(self.check_num):
            check_result, false_num, accuracy = self.check()
            # print
            print(f'check{i}: false_num = {false_num}, accuracy = {accuracy}%')

        # check num
        check_num = self.check_same_num()
        print(f'extract number == nc number: {check_num}')

    def check(self):
        ''' check '''
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

    def check_same_num(self):
        ''' check whether extract num is equal to nc file '''
        ret = len(self.nc_path) == len(self.extract_data)
        return ret

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
    extract_data_path = 'H:/research/flash_drough/GLDAS_Catchment/SoilMoist_RZ_tavg_19480101_20141230.npy'
    source_data_path = 'D:/GLADS/daily_data'
    coord_path = "H:/GIS/Flash_drought/coord.txt"
    variable_name = 'SoilMoist_RZ_tavg'
    # r = re.compile(r"\d{8}\.\d{4}")
    r = re.compile(r'\d{8}')
    ce = CheckExtract(extract_data_path=extract_data_path, source_data_path=source_data_path, coord_path=coord_path,
                      variable_name=variable_name, r=r, precision=3, check_num=1)
    ce.run()