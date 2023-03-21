import math

import numpy as np
import modelling.plot_data as plotter
from scipy.integrate import odeint

from modelling import plot_data


def SIR_model(N, beta, D, initial_I, time_span, to_file):
    gamma = 1.0 / D
    S0, I0, R0 = N - initial_I, initial_I, 0  # initial conditions

    t = np.arange(time_span) # Grid of time points (in days)
    y0 = S0, I0, R0 # Initial conditions vector

    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv_SIR, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    plotter.plot_SIR(t, S, I, R, to_file)


def deriv_SIR(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def SIS_model(N, beta, D, initial_I, time_span, to_file):
    gamma = 1.0 / D
    S0, I0 = N - initial_I, initial_I  # initial conditions

    # Season 1
    t = np.arange(time_span) # Grid of time points (in days)
    y0 = S0, I0 # Initial conditions vector

    # Integrate the SIS equations over the time grid, t.
    ret = odeint(deriv_SIS, y0, t, args=(N, beta, gamma))
    S, I = ret.T

    plotter.plot_SIS(t, S, I, N, to_file)


def SIS_bi_model(N, beta1, D1, beta2, D2, initial_I, time_span, to_file):
    gamma1 = 1.0 / D1
    S0, I0 = N - initial_I, initial_I  # initial conditions

    # Season 1
    t = np.arange(time_span / 2)  # Grid of time points (in days)
    y0 = S0, I0  # Initial conditions vector

    # Integrate the SIS equations over the time grid, t.
    ret = odeint(deriv_SIS, y0, t, args=(N, beta1, gamma1))
    S, I = ret.T

    # Season 2
    gamma2 = 1.0 / D2
    t2 = np.arange(time_span / 2) + time_span / 2  # Grid of time points (in days)
    y2 = S[-1], I[-1]  # Initial conditions vector

    # Integrate the SIS equations over the time grid, t.
    ret = odeint(deriv_SIS, y2, t2, args=(N, beta2, gamma2))
    S2, I2 = ret.T

    t = np.append(t, t2)
    S = np.append(S, S2)
    I = np.append(I, I2)

    plotter.plot_SIS(t, S, I, to_file)


def deriv_SIS(y, t, N, beta, gamma):
    S, I = y
    dSdt = (gamma * I) - (beta * S * I / N)
    dIdt = (beta * S * I / N) - (gamma * I)
    return dSdt, dIdt


def plot_SIS_seasonal_model(N, beta_max, D, initial_I, time_span, to_file):
    t, S, I = SIS_seasonal_model(N, beta_max, D, initial_I, time_span)
    plotter.plot_SIS(t, S, I, N, to_file)


def SIS_seasonal_model(N, beta_max, D, initial_I, time_span):
    gamma = 1.0 / D
    S0, I0 = N - initial_I, initial_I  # initial conditions

    t = np.arange(time_span)  # Grid of time points (in days)
    y0 = S0, I0  # Initial conditions vector

    # Integrate the SIS equations over the time grid, t.
    ret = odeint(deriv_SIS_continuous, y0, t, args=(N, get_beta, beta_max, gamma, time_span))
    S, I = ret.T

    return t, S, I


def deriv_SIS_continuous(y, t, N, beta_function, beta_max, gamma, time_span):
    S, I = y

    # setup sine function for beta
    time = (t / time_span * (2 * np.pi)) - np.pi / 2 # fit one sine wave per year
    beta1 = 0.5 # amplitude
    omega = 1 # frequency of sine wave - 1 per year
    theta = -np.pi / 2 # offset wave in x axis
    beta0 = 0.5 # offset wave in y
    beta = beta_function(beta_max, beta1, omega, time, theta, beta0)

    dSdt = (gamma * I) - (beta * S * I / N)
    dIdt = (beta * S * I / N) - (gamma * I)


    return dSdt, dIdt


def generate_sine(time_span, beta1, omega, theta, beta0):
    time = np.arange(time_span)
    sine = []

    for t in time:
        sine.append(max((beta1 * math.sin((omega * t) + theta) + beta0), 0))

    return sine


def generate_gaussian(time_span, a, t0, sig):
    time = np.arange(time_span)
    gaussian = []

    for t in time:
        gaussian.append(a * math.exp((-(t - t0) ** 2) / (2 * sig ** 2)))

    return gaussian


def generate_repeat_gaussian(time_span, a, t0, sig):
    time = np.arange(time_span)
    gaussian = []

    for t in time:
        gaussian.append(a * math.exp((-((t % 360) - t0) ** 2) / (2 * sig ** 2)))

    return gaussian


def generate_double_repeat_gaussian(time_span, a, t0, sig, a2, t02, sig2):
    time = np.arange(time_span)
    gaussian = []

    for t in time:
        gaussian_1 = a * math.exp((-((t % 360) - t0) ** 2) / (2 * sig ** 2))
        gaussian_2 = a2 * math.exp((-((t % 360) - t02) ** 2) / (2 * sig2 ** 2))

        gaussian.append(gaussian_1 + gaussian_2)

    return gaussian


def SIS_lsf_model(theta, time_span):
    N = 100
    I = 20
    S0, I0 = N - I, I  # initial conditions

    t = np.arange(time_span)  # Grid of time points (in days)
    y0 = S0, I0  # Initial conditions vector

    # Integrate the SIS equations over the time grid, t.
    ret = odeint(deriv_SIS_lsf, y0, t, args=(N, get_beta, theta))
    S, I = ret.T

    I = I/N

    # Take the average of every 30 day period as that month's value
    expected = []
    for day in np.linspace(0, time_span - 30, int(time_span / 30)):
        day = int(day)
        expected.append(np.mean(I[day:day + 29]))

    return np.asarray(expected)


def SIS_lsf_model_gaussian(theta, time_span):
    N = 100
    I = 20
    S0, I0 = N - I, I  # initial conditions

    t = np.arange(time_span)  # Grid of time points (in days)
    y0 = S0, I0  # Initial conditions vector

    # Integrate the SIS equations over the time grid, t.
    ret = odeint(deriv_SIS_lsf_gaussian, y0, t, args=(N, get_beta_gaussian, theta))
    S, I = ret.T

    I = I/N

    # Take the average of every 30 day period as that month's value
    expected = []
    for day in np.linspace(0, time_span - 30, int(time_span / 30)):
        day = int(day)
        expected.append(np.mean(I[day:day + 29]))

    return np.asarray(expected)


def SIS_lsf_model_double_gaussian(theta, time_span):
    N = 100
    I = 20
    S0, I0 = N - I, I  # initial conditions

    t = np.arange(time_span)  # Grid of time points (in days)
    y0 = S0, I0  # Initial conditions vector

    # Integrate the SIS equations over the time grid, t.
    ret = odeint(deriv_SIS_lsf_double_gaussian, y0, t, args=(N, get_beta_double_gaussian, theta))
    S, I = ret.T

    I = I/N

    # Take the average of every 30 day period as that month's value
    expected = []
    for day in np.linspace(0, time_span - 30, int(time_span / 30)):
        day = int(day)
        expected.append(np.mean(I[day:day + 29]))

    return np.asarray(expected)


def deriv_SIS_lsf(y, t, N, beta_function, theta):
    S, I = y
    gamma = 1 / theta[0]

    # setup sine function for beta
    amp = theta[1] # amplitude
    frequency = (2 * np.pi) / 360 # frequency of sine wave - 1 per year
    phase = theta[2] # offset wave in x axis
    beta0 = theta[3] # offset wave in y
    beta = beta_function(amp, frequency, t, phase, beta0)

    recovered = 0
    infected = 0

    if beta > 0:
        infected = (beta * S * I / N)
        recovered = (gamma * I)

    dSdt = recovered - infected
    dIdt = infected - recovered

    return dSdt, dIdt


def deriv_SIS_lsf_gaussian(y, t, N, beta_function, theta):
    S, I = y
    gamma = 1 / theta[0]

    # setup sine function for beta
    a = theta[1] # amplitude
    t0 = theta[2] # offset wave in x axis
    sig = theta[3] # offset wave in y
    beta = beta_function(a, t, t0, sig)

    recovered = 0
    infected = 0

    if beta > 0:
        infected = (beta * S * I / N)
        recovered = (gamma * I)

    dSdt = recovered - infected
    dIdt = infected - recovered

    return dSdt, dIdt


def deriv_SIS_lsf_double_gaussian(y, t, N, beta_function, theta):
    S, I = y
    gamma = 1 / theta[0]

    # setup sine function for beta
    a = theta[1] # amplitude
    t0 = theta[2] # offset wave in x axis
    sig = theta[3] # offset wave in y
    a2 = theta[4] # amplitude
    t02 = theta[5] # offset wave in x axis
    sig2 = theta[6] # offset wave in y
    beta = beta_function(t, a, t0, sig, a2, t02, sig2)

    recovered = 0
    infected = 0

    if beta > 0:
        infected = (beta * S * I / N)
        recovered = (gamma * I)

    dSdt = recovered - infected
    dIdt = infected - recovered

    return dSdt, dIdt


def get_beta(beta1, omega, t, theta, beta0):
    return max((beta1 * math.sin((omega * t) + theta) + beta0), 0)


def get_beta_gaussian(a, t, t0, sig):
    return max(a * math.exp((-((t % 360) - t0) ** 2) / (2 * sig ** 2)), 0)


def get_beta_double_gaussian(t, a, t0, sig, a2, t02, sig2):
    gaussian_1 = a * math.exp((-((t % 360) - t0) ** 2) / (2 * sig ** 2))
    gaussian_2 = a2 * math.exp((-((t % 360) - t02) ** 2) / (2 * sig2 ** 2))

    return max(gaussian_1 + gaussian_2, 0)


class DiseaseModels:
    pass
