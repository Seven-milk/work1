# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com

# extract variable(given region by coord) from .nc4 file
# 1) writing-while-reading to avoid limitation in memory based on write mode = 'a'
# 2) meanwhile, save file to a .npy to reduce file size (compare: .npy : 1022kb vs .txt: 2357kb in all file == 111)
# 3) meanwhile, using multiprocessing to increase efficiency, each cpu write into one file within 'a' mode to avoid
# share memory and keep processing safety, finally, you can combine all file into one .npy file, if .npy file is exist
# in the current path, it will be combined together automatically in the next extract (will be reconized in the
# .combinefiles())

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
                 precision=3, coordsave=False, num_cpu=8, iscombine: bool = True):
        ''' init function
        input: similar with ExtractNcWwrBin
            num_cpu: the number of processes
            iscombine: bool, whether to combine

            note: Process Safety: don't need lock, because each processing handles unique file, there is no sharing
                  memory
        output:
            {variable_name}.bin/.npy [i, j]: i(file number) j(grid point number), bytes file, encoding = UTF8
                to read: x = np.loadtxt('./SoilMoi0_10cm_inst.bin')
                         x = np.load('./SoilMoi0_10cm_inst.npy')
            self.result
            self.coord
            lat_index.txt/lon_index.txt/coord.txt: optional
        '''
        super(ExtractNcWwrBinMp, self).__init__(path, coord_path, variable_name, r, fname, start, end, format,
                                                precision, coordsave)
        self.num_cpu = num_cpu
        self.iscombine = iscombine

    def __call__(self):
        ''' Implement WorkBase.__call__ '''
        self.ret, self.target = self.extract_nc()
        print("start post_process ...")
        self.post_process()
        print("post_process completed")

        print("target:\n", self.target, "\nret:\n", self.ret)

        if self.iscombine == True:
            self.combinefiles()
        print("complete")

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

        # make sure the file is not exist: reason, mode = 'a' -> keep safe
        self.rm_files(rf'{self.fname}*.bin')
        if self.start == "":
            self.rm_files(rf'{self.fname}*.npy')

        # input start/end date and find the index in file_name
        index_start = 0 if self.start == "" else [self.start in name for name in result].index(True)
        index_end = len(result) - 1 if self.end == "" else [self.end in name for name in result].index(True)
        print("start - end: ", self.start, " to ", self.end)

        result = result[index_start: index_end + 1]

        print(f"file number:{len(result)}")

        # read variable based on the lat_index/lon_index in multi-processing
        po = Pool(processes=self.num_cpu)
        section_index = useful_func.divideLen(len(result), self.num_cpu)
        print(f'cpu number = {self.num_cpu}')
        target = []
        for i in range(self.num_cpu):
            target_ = f'cpu{i}: {result[section_index[i]: section_index[i + 1]][0]} to' \
                      f' {result[section_index[i]: section_index[i + 1]][-1]}'
            print(target_)
            target.append(target_)

        ret_po = [po.apply_async(self.read_write, (i, result[section_index[i]: section_index[i + 1]], coord, lat_index,
                                                lon_index, self.r)) for i in range(self.num_cpu)]

        po.close()
        po.join()
        ret = [ret_po[i].get()[0] for i in range(self.num_cpu)]
        return ret, target

    def read_write(self, cpu, result, coord, lat_index, lon_index, r):
        ''' read and write file '''
        start = r.search(result[0])[0]
        end = r.search(result[-1])[0]

        for i in range(len(result)):
            try:
                f = Dataset(result[i], 'r')
            except:
                raise SyntaxError(f'File error: {r.search(result[i])} can not open!')

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
            with open(f'{self.fname}cpu{cpu}_{start}_{end}.bin', mode='ab') as savefile:
                variable.tofile(savefile, format=self.format)

            print(f"cpu{cpu} : {'%.2f' % (i / len(result) * 100)}% - reading and writing file")
            f.close()

        # post process
        print('--------------------------------')
        print(f'cpu{cpu} calculation is complete')
        print('--------------------------------')
        return f'cpu{cpu} calculation is complete'

    def post_process(self):
        ''' override ExtractNcWwrBin.post_process '''
        # files = [file for file in os.listdir() if file.startswith(f'{self.fname}cpu') and file.endswith('.bin')]
        files = glob.glob(rf'{self.fname}cpu*.bin')
        for i in range(len(files)):
            if i == 0:
                # read and reshape
                bin_array = np.fromfile(files[i])
                bin_array = bin_array.reshape((int(len(bin_array) / (len(self.coord) + 1)), len(self.coord) + 1))
            else:
                bin_ = np.fromfile(files[i])
                bin_ = bin_.reshape((int(len(bin_) / (len(self.coord) + 1)), len(self.coord) + 1))
                bin_array = np.vstack((bin_array, bin_))

        # sort file
        bin_array = bin_array[bin_array[:, 0].argsort()]
        start = self.r.search("%.8f" % bin_array[0, 0])[0]
        end = self.r.search("%.8f" % bin_array[-1, 0])[0]

        # save
        if os.path.exists(f'{self.fname}_{start}_{end}.npy'):
            os.remove(f'{self.fname}_{start}_{end}.npy')

        np.save(f'{self.fname}_{start}_{end}', bin_array)

        # rm bin, cache file
        [self.rm_files(rf'{self.fname}cpu*.bin') for i in range(self.num_cpu)]

    def combinefiles(self):
        ''' combine all .npy file into one file '''
        # files = [file for file in os.listdir() if file.startswith(f'{self.fname}') and file.endswith('.npy')]
        files = glob.glob(rf'{self.fname}*.npy')
        if len(files) > 1:
            for i in range(len(files)):
                if i == 0:
                    combine = np.load(files[i])
                else:
                    array_ = np.load(files[i])
                    combine = np.vstack((combine, array_))

                # rm file
                os.remove(files[i])

            # sort
            combine = combine[combine[:, 0].argsort()]
            start = self.r.search("%.8f" % combine[0, 0])[0]
            end = self.r.search("%.8f" % combine[-1, 0])[0]

            # save
            np.save(f'{self.fname}_{start}_{end}', combine)

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


def extractGLDASNOAH():
    path = "E:/GLDAS_NOAH"
    coord_path = "H:/GIS/Flash_drought/coord.txt"
    r = re.compile(r"\d{8}\.\d{4}")
    # r = re.compile(r'\d{8}')
    encmp = ExtractNcWwrBinMp(path, coord_path,
                              "CanopInt_inst",
                              # Tair_f_inst Rainf_f_tavg  ESoil_tavg PotEvap_tavg Wind_f_inst AvgSurfT_inst CanopInt_inst ECanop_tavg
                              start="", end="",
                              r=r, precision=3, num_cpu=8)  # 19480101.0000 19801231.2100 19810101.0000 20141231.2100
    # encmp.overview()
    print(encmp)
    encmp()


if __name__ == "__main__":
    extractGLDASNOAH()