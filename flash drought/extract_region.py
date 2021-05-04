# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# extract a region(Helong) data from original data: such as Drought/FD params, sm, sm_pentad... base on the coord
import os
import pandas as pd
import numpy as np
import warnings
import Workflow

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

    def extract_data(self, coord_region: pd.DataFrame, coord_original: pd.DataFrame, data_original, axis: int = 0):
        ''' extract region data from original data
        input:
            coord_region: pd.DataFrame, region coord, it should have the same format with coord_orginal and be part of
                          coord_original
            coord_orginal: pd.DataFrame, original coord, it should contain columns "lat" & "lon", every in this dataframe
                          is unique value
            data_original: 1D or 2D np.ndarray / pd.Dataframe
            axis: int, 0 or 1, the axis corresponding coord(grid), such as data_original(n*m), n: time, m: coord(grid)
                 -> axis=1, default = 0
        out:
            ret: result, which has the same dimension with data_original, but the "coord" dimension has been extracted
                (len  = len(coord_region)), if data_original is np.ndarray / pd.Dataframe, it will be np.ndarray /
                pd.Dataframe

        note:
            1) index_col=0: pd.read_excel(os.path.join(home, "5.Analysis_spatial/static_params.xlsx"), index_col=0)
        '''
        index = self.cal_index(coord_region, coord_original)
        self.df_true = isinstance(data_original, pd.DataFrame)

        if self.df_true:
            columns = data_original.columns
            data_original = data_original.to_numpy()

        if len(data_original.shape) == 1:
            ret = data_original[index]
        elif len(data_original.shape) == 2:
            if axis == 0:
                ret = data_original[index, :]
            elif axis == 1:
                data_ = data_original.T
                ret = data_[index, :].T
            else:
                raise ValueError("axis: int, 0 or 1, but others is given")
        else:
            raise IndexError("too many indices for data_original array: array is 1/2-dimensional, but more were given")

        if self.df_true:
            ret = pd.DataFrame(ret, columns=columns)

        self.ret = ret

        return ret

    def save(self, path=None):
        ''' save extract data
        input:
            path: save path, if dara_original is pd.Dataframe, path should end with '.xlsx', if it is np.ndarray, it
                  should end with '.txt'
        '''
        if self.df_true:
            if path == None:
                path = 'extract.xlsx'
            self.ret.to_excel(path)
        else:
            if path == None:
                path = 'extract.txt'
            np.savetxt(path, self.ret)


class ExtractRegion(Workflow.WorkBase):
    ''' Work, extract region data from all data, i.e. sm_rz to sm_rz_Helong '''

    def __init__(self, coord_region: pd.DataFrame, coord_original: pd.DataFrame, data_original, axis: int = 0,
                 save_path=None, info=""):
        ''' init function
        input:
            similar with extract_region
            save_path: str, home path to save, if save_path=None(default), do not save
            info: str, informatiom for this Class to print and save in save_path, shouldn't too long
        '''
        self.coord_region = coord_region
        self.coord_original = coord_original
        self.data_original = data_original
        self.axis = axis
        self.save_path = save_path
        self._info = info

    def run(self):
        ''' implement WorkBase.run '''
        extract_region_ = extract_region()
        region_data = extract_region_.extract_data(self.coord_region, self.coord_original, self.data_original,
                                                   self.axis)

        # save result
        if self.save_path != None:
            if extract_region_.df_true:
                suffix = ".xlsx"
            else:
                suffix = ".txt"
            extract_region_.save(os.path.join(self.save_path, f"region_data_{self._info}{suffix}"))

        return region_data

    def __repr__(self):
        return f"This is ExtractRegion, info: {self._info}, extract region data from all data"

    def __str__(self):
        return f"This is ExtractRegion, info: {self._info}, extract region data from all data"


def extract_sm_static_params():
    # original data
    sm_rz = np.loadtxt(data_path, dtype="float", delimiter=" ")
    static_params = pd.read_excel(os.path.join(home, "5.Analysis_spatial/static_params.xlsx"), index_col=0)

    # extract data based on ExtractRegion(WorkFlow)
    ER_sm_rz_Helong = ExtractRegion(coord_Helong, coord, sm_rz, axis=1, save_path=None, info="sm_rz_Helong")  # home
    ER_sm_rz_noHelong = ExtractRegion(coord_noHelong, coord, sm_rz, axis=1, save_path=None, info="sm_rz_noHelong")
    ER_static_params_Helong = ExtractRegion(coord_Helong, coord, static_params, axis=0, save_path=None,
                                            info="static_params_Helong")
    ER_static_params_noHelong = ExtractRegion(coord_noHelong, coord, static_params, axis=0, save_path=None,
                                              info="static_params_noHelong")

    # WF add
    WF_sm_static_params = Workflow.WorkFlow(ER_sm_rz_Helong, ER_sm_rz_noHelong, ER_static_params_Helong,
                                            ER_static_params_noHelong)
    WF_sm_static_params.runflow()
    return WF_sm_static_params

    # extract data based on extract_region
    # extract_region_ = extract_region()
    # sm_rz_Helong = extract_region_.extract_data(coord_Helong, coord, sm_rz, axis=1)
    # sm_rz_noHelong = extract_region_.extract_data(coord_noHelong, coord, sm_rz, axis=1)
    # static_params_Helong = extract_region_.extract_data(coord_Helong, coord, static_params, axis=0)
    # static_params_noHelong = extract_region_.extract_data(coord_noHelong, coord, static_params, axis=0)

    # save data
    # np.savetxt(os.path.join(home, "sm_rz_Helong.txt"), sm_rz_Helong)
    # np.savetxt(os.path.join(home, "sm_rz_noHelong.txt"), sm_rz_noHelong)
    # static_params_Helong.to_excel(os.path.join(home, "static_params_Helong.xlsx"))
    # static_params_noHelong.to_excel(os.path.join(home, "static_params_noHelong.xlsx"))


def extract_Drought_FD_number():
    # original data
    Drought_year_number = pd.read_excel(os.path.join(home, "Drought_year_number.xlsx"), index_col=0)
    FD_year_number = pd.read_excel(os.path.join(home, "FD_year_number.xlsx"), index_col=0)

    # extract data based on ExtractRegion(WorkFlow)
    ER_Drought_year_number_Helong = ExtractRegion(coord_Helong, coord, Drought_year_number, axis=0, save_path=None,
                                                  info="Drought_year_number_Helong")  # home
    ER_Drought_year_number_noHelong = ExtractRegion(coord_noHelong, coord, Drought_year_number, axis=0, save_path=None,
                                                    info="Drought_year_number_noHelong")
    ER_FD_year_number_Helong = ExtractRegion(coord_Helong, coord, FD_year_number, axis=0, save_path=None,
                                             info="FD_year_number_Helong")
    ER_FD_year_number_noHelong = ExtractRegion(coord_noHelong, coord, FD_year_number, axis=0, save_path=None,
                                               info="FD_year_number_noHelong")

    # WF add
    WF_sm_static_params = Workflow.WorkFlow(ER_Drought_year_number_Helong, ER_Drought_year_number_noHelong,
                                            ER_FD_year_number_Helong, ER_FD_year_number_noHelong)
    WF_sm_static_params.runflow()
    return WF_sm_static_params

    # extract data based on extract_region
    # extract_region_ = extract_region()
    # Drought_year_number_Helong = extract_region_.extract_data(coord_Helong, coord, Drought_year_number, axis=0)
    # Drought_year_number_noHelong = extract_region_.extract_data(coord_noHelong, coord, Drought_year_number, axis=0)
    # FD_year_number_Helong = extract_region_.extract_data(coord_Helong, coord, FD_year_number, axis=0)
    # FD_year_number_noHelong = extract_region_.extract_data(coord_noHelong, coord, FD_year_number, axis=0)

    # save data
    # Drought_year_number_Helong.to_excel(os.path.join(home, "Drought_year_number_Helong.xlsx"))
    # Drought_year_number_noHelong.to_excel(os.path.join(home, "Drought_year_number_noHelong.xlsx"))
    # FD_year_number_Helong.to_excel(os.path.join(home, "FD_year_number_Helong.xlsx"))
    # FD_year_number_noHelong.to_excel(os.path.join(home, "FD_year_number_noHelong.xlsx"))


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

    # extract_sm_static_params
    wf_extract_sm_static_param = extract_sm_static_params()

    # extract_Drought_FD_number
    wf_extract_Drought_FD_number = extract_Drought_FD_number()
