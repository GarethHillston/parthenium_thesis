import math
import pickle
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.dates as dates
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.integrate import odeint
import scipy.stats as stats
from scipy.optimize import least_squares

from data_overview import data_stats
from imaging import render, functions, get_data, progressions, indices, classify
import modelling.plot_data as plotter
from modelling import disease_models, plot_data, model_tests

fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'

stats = pickle.load(open('{fp}/parth_presence/prediction_stats.pkl'.format(fp=fp), 'rb'))

render.stack_stats(stats, 'Average Parthenium likelihood prediction per date 2018-2022', 'mean', False, True, '{fp}/ndvi/plots/parth_stats_year_stack_better.png'.format(fp=fp))
# render.stack_stats(stats, 'Average Parthenium likelihood prediction per date 2018-2022', 'mean', False, True, '')
