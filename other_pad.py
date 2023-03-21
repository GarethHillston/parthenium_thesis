import pickle

import xarray as xr
import numpy as np

import utilities
import utilities as util
from data_overview import data_stats
from imaging import render, functions, get_data, progressions, indices, classify
from modelling import model_tests
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from astropy.time import Time
import seaborn as sns
import pandas as pd

from modelling import disease_models, plot_data

fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'

# prec = pd.read_csv('{fp}/precipitation.csv'.format(fp=fp))
# temp = pd.read_csv('{fp}/temperature.csv'.format(fp=fp))
#
# print(prec)
# print(temp)
#
# render.temperature(temp, '', '')
# render.precipitation(prec, '', '')

# park = np.load('{fp}/raw_data/ref_2022-08-02_2022-08-14_mcr.npy'.format(fp=fp), allow_pickle=True)
#
# print(park)




