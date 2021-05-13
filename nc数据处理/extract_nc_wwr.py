# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

# extract variable(given region by coord) from .nc4 file
# 1) writing-while-reading to avoid limitation in memory based on write mode = 'a', it can be continue appending into
# the .bin file, so do not rm it
# 2) meanwhile, save file to a .npy to reduce file size (compare: .npy : 1022kb vs .txt: 2357kb in all file == 111)

import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd
import time
import re
import datetime
import Workflow


class ExtractNcWwrBin(Workflow.WorkBase):
    ''' Work, Extract Nc file writing while reading, save as a Bin file '''
    def __init__(self, path, coord_path, variable_name, r, fname=None, start="", end="", format="%s",
                 precision=3, coordsave=False):
        ''' init function
        input:
            path: path of the source nc file
            coord_path: path of the coord extracted by fishnet: OID_, lon, lat
            variable_name: name of the variable need to read
            precision: the minimum precision of lat/lon, to match the lat/lon of source nc file
            r: <class 're.Pattern'>, regular experssion to identify time, use re.compile(r"...") to build it
                e.g. 19980101 - r = re.compile(r"\d{8}")
            fname: filename to save, default == None(variable_name)
            coordsave: whether save the lat_index/ lon_index/ coord
            start: control the start file to extract, its format is similar as r, default="", namely start=0 (include)
                e.g. start = "20001021.0600"
            end: similar with start, control the end file to extract (include)

            format: save format, such as '.2f' to controld the decimal digits, default="%s" in
                    np.tofile()

        output:
            {variable_name} [i, j]: i(file number) j(grid point number), bytes file, encoding = UTF8
                to read: x = np.loadtxt('H:/research/flash_drough/code/nc数据处理/SoilMoi0_10cm_inst')
            self.result
            self.coord
            lat_index.txt/lon_index.txt
            coord.txt
        '''

        self.path = path
        self.coord_path = coord_path
        self.variable_name = variable_name
        self.r = r
        self.format = format
        self.fname = fname
        self.start = start
        self.end = end
        self.precision = precision
        self.coordsave = coordsave
        self._info = f"{self.variable_name}_{self.start}_{self.end}"

        if self.fname == None:
            self.fname = self.variable_name

        # read coord and load file in path
        self.coord = pd.read_csv(self.coord_path, sep=",")  # read coord(extract by fishnet)
        self.coord = self.coord.round(self.precision)  # coord precision correlating with .nc file lat/lon
        self.result = [self.path + "/" + d for d in os.listdir(self.path) if d[-4:] == ".nc4"]
        self.result.sort(key=lambda x: float(r.search(x)[0]))  # sort result

    def run(self):
        ''' Implement WorkBase.run '''
        self.extract_nc()

    def extract_nc(self):
        """ extract variable(given region by coord) from .nc file
        output:
            f'{self.fname}.npy': binary file (smaller), read by np.load(path)
            f'{self.fname}.bin': binary file, before post_process, it's a cache file, do not remove
        """
        coord = self.coord
        result = self.result
        print(f"variable:{self.variable_name}")
        print(f"grid point number:{len(coord)}")

        # calculate the index of lat/lon in coord from source nc file
        lat_index, lon_index = self.cal_index(result[0], coord)

        # save lat_index/lon_index and coord
        if self.coordsave == True:
            np.savetxt('lat_index.txt', lat_index, delimiter=' ')
            np.savetxt('lon_index.txt', lon_index, delimiter=' ')
            coord.to_csv("coord.txt")

        # make sure the file is not exist: reason, mode = 'a'
        if self.start == "":
            self.rm_files(f'{self.fname}.bin')

        # input start/end date and find the index in file_name
        index_start = 0 if self.start == "" else [self.start in name for name in result].index(True)
        index_end = 0 if self.end == "" else [self.end in name for name in result].index(True)
        print("start - end: ", self.start, " to ", self.end)

        result = result[index_start: index_end + 1]
        print(f"file number:{len(result)}")

        # read variable based on the lat_index/lon_index
        for i in range(len(result)):
            f = Dataset(result[i], 'r')
            variable = np.zeros((len(coord) + 1))
            # Dataset.set_auto_mask(f, False) # if there no mask value, open to improve speed
            variable[0] = float(r.search(result[i])[0])

            # re: the number depend on the nc file name(daily=8, month=6)
            for j in range(len(coord)):
                variable[j + 1] = f.variables[self.variable_name][0, lat_index[j], lon_index[j]]
                # require: nc file only have three dimension
                # f.variables['Rainf_f_tavg'][0, lat_index_lp, lon_index_lp]is a mistake, we only need the file
                # that lat/lon corssed (1057) rather than meshgrid(lat, lon) (1057*1057)

            # save
            with open(f'{self.fname}.bin', mode='ab') as savefile:
                variable.tofile(savefile, format=self.format)

            print(f"complete reading and writing file:{i}")
            f.close()

        # post process reshape and sort file
        print("reshape and sort file...")
        self.post_process()
        print("Complete")

    def post_process(self):
        ''' reshape and sort by time of the binary file '''
        bin_array = np.fromfile(os.path.join(os.getcwd(), f'{self.fname}.bin'))
        bin_array = bin_array.reshape((int(len(bin_array) / (len(self.coord) + 1)), len(self.coord) + 1))
        bin_array = bin_array[bin_array[:, 0].argsort()]

        # save
        if os.path.exists(f'{self.fname}'):
            os.remove(f'{self.fname}')
        np.save(f'{self.fname}', bin_array)

    def rm_files(self, r):
        ''' remove files in this fname
        input:
            r: regular expression for files to rm
        '''
        if os.path.exists(r):
            os.remove(r)

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
            lat_index.append(np.where(lat == coord["lat"][j])[0][0])
            lon_index.append(np.where(lon == coord["lon"][j])[0][0])
        f.close()
        return lat_index, lon_index

    def overview(self):
        # overview of the nc file
        result = [self.path + "/" + d for d in os.listdir(self.path) if d[-4:] == ".nc4"]
        rootgrp = Dataset(result[0], "r")
        print('****************************')
        print(f"number of nc file:{len(result)}")
        print('****************************')
        print(f"variable key:{rootgrp.variables.keys()}")
        print('****************************')
        print(f"rootgrp:{rootgrp}")
        print('****************************')
        print(f"lat:{rootgrp.variables['lat'][:]}")
        print('****************************')
        print(f"lon:{rootgrp.variables['lon'][:]}")
        print(f"variable:{rootgrp.variables}")
        print('****************************')
        variable_name = input("variable name:")  # if you want to see the variable, input its name here
        while variable_name != "":
            print('****************************')
            print(f"variable:{rootgrp.variables[variable_name]}")
            variable_name = input("variable name:")  # if you want to quit, input enter here
        rootgrp.close()

    def __repr__(self):
        return f"This is ExtractNcWwrBin, info: {self._info}, extract variable(given region by coord) from .nc file" \
               f" into .npy file"

    def __str__(self):
        return f"This is ExtractNcWwrBin, info: {self._info}, extract variable(given region by coord) from .nc file" \
               f" into .npy file"


class ExtractNcWwrStr(ExtractNcWwrBin):
    ''' Work, Extract Nc file writing while reading, save as a Str file '''
    def __init__(self, path, coord_path, variable_name, r, fname=None, start="", end="", format="%s",
                 func=lambda x: str(float(x)), precision=3, coordsave=False):
        '''
        init function
        input: similar with ExtractNcWwrBin
            func: extract_nc_str, callable, handle the v_ and return a str, variable.append(str(func(v_))),
                            default=lambda x: str(float(x))
                e.g. lambda x: format(x, '.2f') to change the decimal digits
        '''

        super(ExtractNcWwrStr, self).__init__(path, coord_path, variable_name, r, fname, start, end, format,
                                              precision, coordsave)
        self.func = func

        if callable(self.func) != True:
            raise ValueError("func should be a callable function")

    def extract_nc(self):
        """ extract variable(given region by coord) from .nc file
        output:
            f'{self.fname}.txt': str file (bigger) , read by np.loadtxt(path)
        """
        coord = self.coord
        result = self.result
        print(f"variable:{self.variable_name}")
        print(f"grid point number:{len(coord)}")

        # calculate the index of lat/lon in coord from source nc file
        lat_index, lon_index = self.cal_index(result[0], coord)

        # save lat_index/lon_index and coord
        if self.coordsave == True:
            np.savetxt('lat_index.txt', lat_index, delimiter=' ')
            np.savetxt('lon_index.txt', lon_index, delimiter=' ')
            coord.to_csv("coord.txt")

        # make sure the file is not exist: reason, mode = 'a'
        if self.start == "":
            self.rm_files(f'{self.fname}.txt')

        # input start/end date and find the index in file_name
        index_start = 0 if self.start == "" else [self.start in name for name in result].index(True)
        index_end = 0 if self.end == "" else [self.end in name for name in result].index(True)
        print("start - end: ", self.start, " to ", self.end)

        result = result[index_start: index_end + 1]
        print(f"file number:{len(result)}")

        # read variable based on the lat_index/lon_index
        for i in range(len(result)):
            f = Dataset(result[i], 'r')
            variable = []
            # Dataset.set_auto_mask(f, False) # if there no mask value, open to improve speed
            variable.append(str(r.search(result[i])[0]))

            # re: the number depend on the nc file name(daily=8, month=6)
            for j in range(len(coord)):
                v_ = f.variables[self.variable_name][0, lat_index[j], lon_index[j]]
                variable.append(self.func(v_))
                # require: nc file only have three dimension
                # f.variables['Rainf_f_tavg'][0, lat_index_lp, lon_index_lp]is a mistake, we only need the file
                # that lat/lon corssed (1057) rather than meshgrid(lat, lon) (1057*1057)

            # save
            with open(f'{self.fname}.txt', mode='a') as savefile:
                savefile.write(" ".join(variable) + "\n")

            print(f"complete reading and writing file:{i}")
            f.close()

        # str_sort_by_time
        self.post_process()

    def post_process(self):
        ''' sort result by time '''
        variable = np.loadtxt(f'{self.variable_name}.txt')
        variable = variable[variable[:, 0].argsort()]

        with open(f'{self.fname}', mode='ab') as savefile:
            savefile.write(bytes(" ".join(variable) + "\n", encoding='UTF8'))
        np.savetxt(f'{self.variable_name}', variable, delimiter=' ')

    def __repr__(self):
        return f"This is ExtractNcWwrStr, info: {self._info}, extract variable(given region by coord) from .nc file" \
               f" into .txt file"

    def __str__(self):
        return f"This is ExtractNcWwrStr, info: {self._info}, extract variable(given region by coord) from .nc file" \
               f" into .txt file"


if __name__ == "__main__":
    # path = "H:/data_zxd/GLDAS_test"
    # coord_path = "H:/GIS/Flash_drought/coord.txt"
    # r = re.compile(r"\d{8}")
    # start = time.time()
    # extract_nc(path, coord_path, "SoilMoist_RZ_tavg", r=r, precision=3)
    # end = time.time()
    # print(end-start)
    path = "D:/GLDAS_NOAH"
    coord_path = "H:/GIS/Flash_drought/coord.txt"
    r = re.compile(r"\d{8}\.\d{4}")
    enc = ExtractNcWwrBin(path, coord_path, "SoilMoi0_10cm_inst", start="19570101.0300", end="19700101.0000", r=r,
                          precision=3)
    enc.overview()
    print(enc)
    enc.run()
