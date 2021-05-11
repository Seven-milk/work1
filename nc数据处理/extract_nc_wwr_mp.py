# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

# extract variable(given region by coord) from .nc4 file
# 1) writing-while-reading to avoid limitation in memory based on write mode = 'a'
# 2) meanwhile, save file to a .npy to reduce file size (compare: .npy : 1022kb vs .txt: 2357kb in all file == 111)
# 3) meanwhile, using multiprocessing to increase efficiency, each cpu write into one file within 'a' mode to avoid
# share memory and keep processing safety, finally, you can combine all file into one .npy file

import numpy as np
from netCDF4 import Dataset
import os
import pandas as pd
# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import multiprocessing
import re
import time
import extract_nc_wwr
import useful_func
import glob


class ExtractNcWwrBinMp(extract_nc_wwr.ExtractNcWwrBin):
    ''' Work, Extract Nc file writing while reading in multi-processing, save as a Bin file '''

    def __init__(self, path, coord_path, variable_name, r, fname=None, start="", end="", format="%s",
                 precision=3, coordsave=False, num_cpu=4, direction="v"):
        ''' init function
        input: similar with ExtractNcWwrBin
            num_cpu: the number of processes
            direction: "v"(vstack) or "h"(hstack), the combine direction, if you do not want to combine .npy files,
                        set direction==None

            note: Process Safety: don't need lock, because each processing handles unique file, there is no sharing
                  memory
        output:

        '''
        super(ExtractNcWwrBinMp, self).__init__(path, coord_path, variable_name, r, fname, start, end, format,
                                                precision, coordsave)
        self.num_cpu = num_cpu
        if direction != "v" and direction != "h" and direction != None:
            raise ValueError("direction should be 'v' or 'h' or None")
        self.direction = direction

    def run(self):
        ''' Implement WorkBase.run '''
        self.extract_nc()
        if self.direction != None:
            self.combinefiles(self.direction)

    def extract_nc(self):
        ''' override ExtractNcWwrBin.extract_nc '''
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
            self.rm_files(rf'{self.fname}*.bin')
            self.rm_files(rf'{self.fname}*.npy')

        # input start/end date and find the index in file_name
        index_start = 0 if self.start == "" else [self.start in name for name in result].index(True)
        index_end = 0 if self.end == "" else [self.end in name for name in result].index(True)
        print("start - end: ", self.start, " to ", self.end)

        result = result[index_start: index_end + 1]
        print(f"file number:{len(result)}")

        # read variable based on the lat_index/lon_index in multi-processing
        po = Pool(processes=self.num_cpu)
        section_index = useful_func.divideLen(len(result), self.num_cpu)
        _ = [po.apply_async(self.read_write, (i, result[section_index[i]: section_index[i + 1]], coord, lat_index,
                                                lon_index, r)) for i in range(self.num_cpu)]
        po.close()
        po.join()

    def read_write(self, cpu, result, coord, lat_index, lon_index, r):
        ''' read and write file '''
        start = r.search(result[0])[0]
        end = r.search(result[-1])[0]
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
            with open(f'{self.fname}cpu{cpu}.bin', mode='ab') as savefile:
                variable.tofile(savefile, format=self.format)

            print(f"cpu{cpu} : complete reading and writing file:{i}")
            f.close()

        # post process
        self.post_process(cpu, start, end)

    def post_process(self, cpu, start, end):
        ''' override ExtractNcWwrBin.post_process '''
        # reshape and sort file
        bin_array = np.fromfile(os.path.join(os.getcwd(), f'{self.fname}cpu{cpu}.bin'))
        bin_array = bin_array.reshape((int(len(bin_array) / (len(self.coord) + 1)), len(self.coord) + 1))
        bin_array = bin_array[bin_array[:, 0].argsort()]

        # save
        if os.path.exists(f'{self.fname}_{start}_{end}'):
            os.remove(f'{self.fname}_{start}_{end}')

        np.save(f'{self.fname}_{start}_{end}', bin_array)

        # rm bin
        self.rm_files(rf'{self.fname}cpu{cpu}.bin')

    def combinefiles(self, direction):
        ''' combine all .npy file into one file '''
        files = glob.glob(rf'{self.fname}*.npy')
        if len(files) > 0:
            for i in range(len(files)):
                if i == 0:
                    combine = np.load(files[i])
                else:
                    array_ = np.load(files[i])
                    if direction == "v":
                        combine = np.vstack((combine, array_))
                    elif direction == "h":
                        combine = np.hstack((combine, array_))

                # rm file
                os.remove(files[i])

            # sort
            combine = combine[combine[:, 0].argsort()]

            # save
            np.save(f'{self.fname}_{self.start}_{self.end}', combine)

    def rm_files(self, rf):
        ''' override ExtractNcWwrBin.rm_bin
        input:
            rf: regular expression for files to rm, e.g. rf = rf'{self.fname}*.bin'
        '''
        files = glob.glob(rf)
        if len(files) > 0:
            for file in files:
                os.remove(file)

    def __repr__(self):
        return f"This is ExtractNcWwrBinMp, info: {self._info}, extract variable(given region by coord) from .nc file" \
               f" into .npy file by multiprocessing"

    def __str__(self):
        return f"This is ExtractNcWwrBinMp, info: {self._info}, extract variable(given region by coord) from .nc file" \
               f" into .npy file by multiprocessing"


if __name__ == "__main__":
    path = "D:/GLDAS_NOAH"
    coord_path = "H:/GIS/Flash_drought/coord.txt"
    r = re.compile(r"\d{8}\.\d{4}")
    encmp = ExtractNcWwrBinMp(path, coord_path, "SoilMoi0_10cm_inst", start="", end="19480201.0000", r=r, precision=3,
                              num_cpu=8)
    # encmp.overview()
    print(encmp)
    encmp.run()
