import numpy as np

from imaging import render
from modelling import simulate

matrix = np.load('test/model/transition_matrix.npy', allow_pickle=True)
start_state = np.load('test/predictions/binaries/binary_prediction_2018-12-27.npy', allow_pickle=True)
# land_classes = np.load('lulc_parth_combined_model/19_20_rawalpindi_land_classes.npy', allow_pickle=True)
# land_classes = land_classes[0]

# lulc_proportions = np.load('lulc_parth_combined_model/parth_lulc_proportions_19_20.npy', allow_pickle=True)
# lulc_proportions = [0.8, 0.7, 0.3, 0.2, 0.1, 0.1, 0, 0]

simulate.markov(start_state, 4, matrix, 'test/model/2018_2021_output')
