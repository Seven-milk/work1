# code: utf-8
# author: "Xudong Zheng" 
# email: Z786909151@163.com
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': 'total_precipitation',
        'year': '2020',
        'month': '07',
        'day': '13',
        'time': '00:00',
        'area': [
            60, 70, 15,
            140,
        ],
    },
    'download.grib')