# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# extract a region(Helong) data from original data: such as Drought/FD params, sm, sm_pentad... base on the coord
import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# cal the coord_region index in coord_original
class index_coord:
    ''' class to cal the indexes of coord_region's each row in coord_original '''

    def cal_index(self, coord_region: pd.DataFrame, coord_original: pd.DataFrame):
        ''' cal the coord_region index belong coord_original array to extract region data from original data
        coord_original                    coord_region
           lat  lon                          lat  lon
        0   1     1                       0   2    2  ——> index in coord_original = 1
        1   2     2                       1           ——> index ....
        2   3     3                       2           ...
        ...                               ...         ...
        100                               30          ...
                                                          output: index
        note: this function is based on a dependency between coord_region and coord_original, it should exists!

        input:
            coord_region: pd.DataFrame, region coord, it should have the same format with coord_orginal and be part of
                          coord_original
            coord_orginal: pd.DataFrame, original coord, it should contain columns "lat" & "lon", every in this dataframe
                          is unique value
        output:
            index: np.ndarray, shape = (len(coord_region), ), the indexes of coord_region's each row in coord_original
        '''
        index = np.zeros((len(coord_region),), dtype=int)
        for i in range(len(coord_region)):
            index[i] = int(coord_original[(coord_original.lat == coord_region.lat[i])]
                           [(coord_original.lon == coord_region.lon[i])].index.values[0])
        return index


class extract_region(index_coord):
    ''' class to extract region data from original data, inherit from index_coord '''

    def extract_data(self, coord_region: pd.DataFrame, coord_original: pd.DataFrame, data_original: np.ndarray,
                     axis: int = 0):
        ''' extract region data from original data
        input:
            coord_region: pd.DataFrame, region coord, it should have the same format with coord_orginal and be part of
                          coord_original
            coord_orginal: pd.DataFrame, original coord, it should contain columns "lat" & "lon", every in this dataframe
                          is unique value
            data_original: 1D or 2D np.ndarray
            axis: int, 0 or 1, the axis corresponding coord(grid), such as data_original(n*m), n: time, m: coord(grid)
                 -> axis=1, default = 0
        out:
            ret: result, which has the same dimension with data_original, but the "coord" dimension has been extracted
                (len  = len(coord_region))
        '''
        index = self.cal_index(coord_region, coord_original)
        if len(data_original.shape) == 1:
            ret = data_original[index]
            return ret
        elif len(data_original.shape) == 2:
            if axis == 0:
                ret = data_original[index, :]
                return ret
            elif axis == 1:
                data_ = data_original.T
                ret = data_[index, :].T
                return ret
            else:
                raise ValueError("axis: int, 0 or 1, but others is given")
        else:
            raise IndexError("too many indices for data_original array: array is 1/2-dimensional, but more were given")


if __name__ == "__main__":
    # general set
    root = "H"
    home = f"{root}:/research/flash_drough/"
    coord_path = os.path.join(home, "coord.txt")
    coord_Helong_path = os.path.join(home, "coord_Helong.txt")
    coord_noHelong_path = os.path.join(home, "coord_no_Helong.txt")
    data_path = os.path.join(home, "GLDAS_Catchment/SoilMoist_RZ_tavg.txt")

    coord = pd.read_csv(coord_path, sep=",")
    coord_Helong = pd.read_csv(coord_Helong_path, sep=",")
    coord_noHelong = pd.read_csv(coord_noHelong_path, sep=",")

    # cal index
    index_coord_ = index_coord()
    index_Helong = index_coord_.cal_index(coord_Helong, coord)
    index_noHelong = index_coord_.cal_index(coord_noHelong, coord)

    # original data
    sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
    static_params = pd.read_excel(os.path.join(home, "5.Analysis_spatial/static_params.xlsx"),
                                  index_col=0)
    static_params_array = static_params.to_numpy()

    # extract data
    extract_region_ = extract_region()
    sm_rz_Helong = extract_region_.extract_data(coord_Helong, coord, sm_rz, axis=1)
    sm_rz_noHelong = extract_region_.extract_data(coord_noHelong, coord, sm_rz, axis=1)
    static_params_Helong = extract_region_.extract_data(coord_Helong, coord, static_params_array, axis=0)
    static_params_noHelong = extract_region_.extract_data(coord_noHelong, coord, static_params_array, axis=0)

    # pd.DataFrame
    static_params_Helong = pd.DataFrame(static_params_Helong, columns=static_params.columns)
    static_params_noHelong = pd.DataFrame(static_params_noHelong, columns=static_params.columns)

    # save data
    # np.savetxt(os.path.join(home, "sm_rz_Helong.txt"), sm_rz_Helong)
    # np.savetxt(os.path.join(home, "sm_rz_noHelong.txt"), sm_rz_noHelong)
    # static_params_Helong.to_excel(os.path.join(home, "static_params_Helong.xlsx"))
    # static_params_noHelong.to_excel(os.path.join(home, "static_params_noHelong.xlsx"))
