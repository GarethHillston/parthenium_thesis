import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares
import numpy as np
from modelling import plot_data, disease_models


def least_square_optimisation():
    fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'

    observed = np.load('{fp}/avg_monthly_parth_cover_0.5_10pc_cloud.npy'.format(fp=fp), allow_pickle=True)
    observed = np.tile(observed, 10)

    time_span = 3600
    symptomatic_period = 210  # D

    amp = 0  # amplitude
    phase = np.pi / 2  # offset wave in x axis
    beta0 = 0.05  # offset wave in y

    theta0 = [symptomatic_period, amp, phase, beta0]

    # define fitting bounds
    D_down = 0
    D_up = np.inf
    amp_down = 0
    amp_up = np.inf
    phase_down = -np.pi
    phase_up = np.pi
    beta0_down = 0
    beta0_up = np.inf

    bounds = (
        [D_down, amp_down, phase_down, beta0_down],
        [D_up, amp_up, phase_up, beta0_up]
    )

    def fun(theta):
        return disease_models.SIS_lsf_model(theta, time_span) - observed

    results = least_squares(fun, theta0, bounds=bounds)

    print("Optimised free parameters:")
    print("D: {}".format(results["x"][0]))
    print("amp: {}".format(results["x"][1]))
    print("phase: {}".format(results["x"][2]))
    print("beta0: {}".format(results["x"][3]))

    expected = disease_models.SIS_lsf_model(results["x"], time_span)

    # plot optimised model
    plot_data.optimise_model(expected, observed,
                             'Simulated Versus Average Actual Parthenium Population Over 10 Years',
                             # '{fp}/plots/lsf_sims/best_phase_10_year_lsf.png'.format(fp=fp))
                             '')

    amp = results["x"][1]  # amplitude
    frequency = (2 * np.pi) / 360  # frequency of sine wave - 1 per year
    phase = results["x"][2]  # offset wave in x axis
    beta0 = results["x"][3]  # offset wave in y
    sine = disease_models.generate_sine(time_span, amp, frequency, phase, beta0)

    # plot sine
    plot_data.plot_sine(sine,
                        'Beta function output',
                        # '{fp}/plots/lsf_sims/best_phase_10_year_beta.png'.format(fp=fp))
                        '')


def least_square_optimisation_gaussian():
    fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'

    observed = np.load('{fp}/avg_monthly_parth_cover_0.5_10pc_cloud.npy'.format(fp=fp), allow_pickle=True)
    observed = np.tile(observed, 10)

    time_span = 3600

    D = 231

    a = 0.0696
    t0 = 284
    sig = 13.5

    theta0 = [D, a, t0, sig] # <- parameters

    # Define fitting bounds
    D_down = 0
    D_up = np.inf
    a_down = 0
    a_up = np.inf
    t0_down = 0
    t0_up = 360
    sig_down = 0
    sig_up = 360

    bounds = (
        [D_down, a_down, t0_down, sig_down], # <- lower bounds
        [D_up, a_up, t0_up, sig_up] # <- upper bounds
    )

    def fun(theta):
        return disease_models.SIS_lsf_model_gaussian(theta, time_span) - observed

    results = least_squares(fun, theta0, bounds=bounds)

    print("Optimised free parameters:")
    print("D = {}".format(results["x"][0]))
    print("a = {}".format(results["x"][1]))
    print("t0 = {}".format(results["x"][2]))
    print("sig = {}".format(results["x"][3]))

    expected = disease_models.SIS_lsf_model_gaussian(results["x"], time_span)

    # plot optimised model
    plot_data.optimise_model(expected, observed,
                             'Simulated Versus Average Actual Parthenium Population Over 10 Years',
                             '{fp}/plots/lsf_sims/gaussian_10_year_lsf.png'.format(fp=fp))
                             # '')

    a = results["x"][1]  # amplitude
    t0 = results["x"][2]  # median
    sig = results["x"][3]  # standard deviation

    gaussian = disease_models.generate_repeat_gaussian(time_span, a, t0, sig)

    # plot sine
    plot_data.plot_gaussian(gaussian,
                        'Beta function output',
                        '{fp}/plots/lsf_sims/gaussian_10_year_beta.png'.format(fp=fp))
                        # '')


def least_square_optimisation_double_gaussian():
    fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'

    observed = np.load('{fp}/avg_monthly_parth_cover_0.5_10pc_cloud.npy'.format(fp=fp), allow_pickle=True)
    observed = np.tile(observed, 10)

    time_span = 3600

    D = 231

    a = 1
    t0 = 180
    sig = 90

    a2 = 1
    t02 = 180
    sig2 = 90

    theta0 = [D, a, t0, sig, a2, t02, sig2] # <- parameters

    # Define fitting bounds
    D_down = 0
    D_up = np.inf
    a_down = 0
    a_up = np.inf
    t0_down = 0
    t0_up = 360
    sig_down = 0
    sig_up = 360

    bounds = (
        [D_down, a_down, t0_down, sig_down, a_down, t0_down, sig_down], # <- lower bounds
        [D_up, a_up, t0_up, sig_up, a_up, t0_up, sig_up] # <- upper bounds
    )

    def fun(theta):
        return disease_models.SIS_lsf_model_double_gaussian(theta, time_span) - observed

    results = least_squares(fun, theta0, bounds=bounds)

    print("Optimised free parameters:")
    print("D = {}".format(results["x"][0]))
    print("a = {}".format(results["x"][1]))
    print("t0 = {}".format(results["x"][2]))
    print("sig = {}".format(results["x"][3]))
    print("a2 = {}".format(results["x"][4]))
    print("t02 = {}".format(results["x"][5]))
    print("sig2 = {}".format(results["x"][6]))

    expected = disease_models.SIS_lsf_model_double_gaussian(results["x"], time_span)

    # plot optimised model
    plot_data.optimise_model(expected, observed,
                             'Simulated Versus Average Actual Parthenium Population Over 10 Years',
                             '{fp}/plots/lsf_sims/double_gaussian_10_year_lsf.png'.format(fp=fp))
                             # '')

    a = results["x"][1]  # amplitude
    t0 = results["x"][2]  # median
    sig = results["x"][3]  # standard deviation
    a2 = results["x"][4]  # amplitude
    t02 = results["x"][5]  # median
    sig2 = results["x"][6]  # standard deviation

    gaussian = disease_models.generate_double_repeat_gaussian(time_span, a, t0, sig, a2, t02, sig2)

    # plot sine
    plot_data.plot_gaussian(gaussian,
                        'Beta function output',
                        '{fp}/plots/lsf_sims/double_gaussian_10_year_beta.png'.format(fp=fp))
                        # '')


class ModelTests:
    pass
