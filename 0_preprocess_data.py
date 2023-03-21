import numpy as np
import rasterio
from rasterio.plot import show
from rasterio.enums import Resampling

date_list = ['2022-08-02']
file_dates = []
for date in date_list:
    file_dates.append(date.replace('-', ''))

reflectance_series = {}
bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

for i in range(0, len(date_list)):
    reflectance = {}
    with rasterio.open("../download_data/{year}/{date}_MCR.tif".format(year='Manchester', date=file_dates[i])) as sr_data:
        for j in range(sr_data.count):
            reflectance[bands[j]] = sr_data.read(j + 1)

    reflectance_series[date_list[i]] = reflectance

shape = np.shape(reflectance_series[date_list[0]][bands[0]])
# scl_series = {}
#
# for i in range(0, len(date_list)):
#     with rasterio.open("../download_data/{year}/{date}_SCL.tif".format(year=date_list[0].split('-')[0], date=file_dates[i])) as scl_data:
#         scl = scl_data.read(
#             out_shape=(
#                 scl_data.count,
#                 shape[0],
#                 shape[1]
#             ),
#             resampling=Resampling.nearest
#         )
#         scl = scl[0]
#
#     scl_series[date_list[i]] = scl
#
# print(np.shape(scl_series[date_list[0]]))
print(np.shape(reflectance_series[date_list[0]][bands[0]]))

np.save('data_overview/raw_data/ref_{start}_{end}_mcr.npy'.format(start=date_list[0], end=date_list[-1]), reflectance_series)
# np.save('data_overview/raw_data/scl_{start}_{end}.npy'.format(start=date_list[0], end=date_list[-1]), scl_series)
