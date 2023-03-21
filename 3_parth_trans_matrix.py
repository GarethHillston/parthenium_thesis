import numpy as np
import xarray as xr
import utilities as util

from modelling import transition_matrix as matrix

# create normal matrix
progressions = np.load('test/predictions/binaries/2018_2021_binary_predictions.npy', allow_pickle=True)
trans_matrix = matrix.create(progressions)
np.save('test/model/transition_matrix.npy', trans_matrix)

# modify for lulc weights
