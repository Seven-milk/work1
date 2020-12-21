# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
# extract data from gldas .nc file through extract_nc.py and ex.extract_nc_mp.py
""" Surface Soil moisture: "SoilMoist_S_tavg"; Root Zone Soil moisture: 'SoilMoist_RZ_tavg';
Profile Soil moisture: 'SoilMoist_P_tavg'; Total precipitation rate: "Rainf_f_tavg"
air_temperature: "Tair_f_tavg"; Direct Evaporation from Bare Soil: "ESoil_tavg"
Plant canopy surface water: 'CanopInt_tavg'; transpiration_flux_from_veg: "TVeg_tavg"
Canopy water evaporation: 'ECanop_tavg'; Evapotranspiration: 'Evap_tavg'
Heat flux: 'Qg_tavg'; """

import extract_nc as ex
import numpy as np
import pandas as pd

""" extarct variable using serial function """
path = "D:\GLADS\daily_data"
coord_path = "H:\GIS\Flash_drought\coord.txt"
coord = pd.read_csv(coord_path, sep=",")
# see extract_nc.overview(path)
# see ex.extract_nc(path, coord_path, 'ESoil_tavg', precision=3)

""" extart variable using parallel function """
# see extarct_nc_mp.py
