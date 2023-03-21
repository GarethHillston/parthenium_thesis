import pickle
import xarray as xr
import numpy as np
import utilities as util
from imaging import get_data, indices, functions, render, classify

filePath = '/scratch/nas_bridle/sentinel/shared/rawalpindi_1.nc'
raw_data = xr.open_dataset(filePath)
date_times = util.get_dates(raw_data)

for i in range(len(date_times)):
    date = date_times[i].split('T')[0]
    binary_prediction = np.load('./progressions/19_20_binaries/bin_predict_' + date + '.npy')
    render.single_plot(binary_prediction, date, ['white', 'mediumseagreen'], 'binaries')
