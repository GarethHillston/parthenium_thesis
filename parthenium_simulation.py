import numpy as np
import utilities
from modelling import transition_matrix, simulate
import xarray as xr

filePath = '/scratch/nas_bridle/sentinel/shared/rawalpindi_1.nc'
raw_data = xr.open_dataset(filePath)

dates = utilities.get_dates(raw_data)

# file_root = './progressions/19_20_binaries/'
# last_prediction = np.load(file_root + 'bin_predict_' + dates[0].split('T')[0] + '.npy', allow_pickle=True)
# matrix_stack = []
#
# for i in range(1, len(dates)):
#     date = dates[i]
#     date_neat = date.split('T')[0]
#     prediction = np.load(file_root + 'bin_predict_' + date_neat + '.npy')
#     matrix_stack.append(transition_matrix.create(np.dstack((last_prediction, prediction))))
#     last_prediction = prediction

matrix_stack = np.load('./matrix_stack.npy')
avg_matrix = np.average(matrix_stack, axis=0)
start_state = np.load('./progressions/19_20_binaries/bin_predict_2020-02-25.npy')
print(avg_matrix)
# simulate.markov(start_state, 50, avg_matrix)
