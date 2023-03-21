import math
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as colours
from matplotlib.colors import ListedColormap
from imaging import render
from matplotlib import dates


# if (out_dir == ''):
#     now = datetime.now().strftime("%d_%m_%y__%H%M%S")
#     self.output_folder = "./modelling/simulations/{dateTime}".format(dateTime=now)
# else:
#     self.output_folder = out_dir
# os.mkdir(self.output_folder)


def replot_binary(self, locations, iteration):
    locations = locations.astype(np.float32)
    np.save('{outputFolder}/sim-{iter:03d}.npy'.format(outputFolder=self.output_folder, iter=iteration), locations)
    figure = plt.figure()
    axes = figure.add_subplot(111)
    cmap = colours.ListedColormap(['white', 'mediumseagreen'])
    axes.imshow(locations, cmap=cmap)
    figure.set_dpi(300)
    plt.savefig('{outputFolder}/sim-{iter:03d}.png'.format(outputFolder=self.output_folder, iter=iteration))
    plt.close()


def replot(self, locations, iteration):
    figure = plt.figure()
    axes = figure.add_subplot(111)
    cMap = colours.ListedColormap(
        {'white', 'navajowhite', 'coral', 'indianred', 'firebrick', 'maroon', 'indigo', 'black'})
    axes.imshow(locations, cmap=cMap)
    figure.set_dpi(300)
    plt.savefig('{outputFolder}/sim-{iter:03d}.png'.format(outputFolder=self.output_folder, iter=iteration))
    plt.close()


def replot_gradient(self, locations, iteration):
    figure = plt.figure()
    axes = figure.add_subplot(111)
    image = axes.pcolormesh(locations, cmap=plt.get_cmap('inferno'), shading='gouraud', vmin=0, vmax=1)
    figure.colorbar(image, ax=axes)
    plt.savefig('{outputFolder}/sim-{iter:03d}.png'.format(outputFolder=self.output_folder, iter=iteration))
    plt.close()


def plot_SIR(t, S, I, R, N, to_file):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')

    ax.set_xlabel('Time (days)')
    ax.set_ylim([0, N])

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    render.output_plot(to_file)


def plot_SIS(t, S, I, N, to_file):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')

    ax.set_xlabel('Time (days)')
    ax.set_ylim([0, N])
    ax.grid(True)

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    render.output_plot(to_file)


def plot_SIS_sine(times, S, I, N, to_file):

    sine = []
    for t in times:
        amp = 1.1223552043992036  # amplitude
        frequency = (2 * np.pi) / 360  # frequency of sine wave - 1 per year
        phase = 2.1326936840487907  # offset wave in x axis
        beta0 = -5.374629809448352  # offset wave in y
        sine.append(max((amp * math.sin((frequency * t) + phase) + beta0), 0))

    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(times, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(times, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(times, sine, 'g', alpha=0.7, linewidth=2, label='Beta')

    ax.set_xlabel('Time (days)')
    # ax.set_ylim([0, N])
    ax.grid(True)

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    render.output_plot(to_file)


def plot_sine(sine, title, to_file):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(np.arange(len(sine)), sine, 'g', alpha=0.7, linewidth=2, label='Beta')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Amplitude')
    ax.title.set_text(title)
    # ax.set_ylim([0, 1])
    ax.grid(True)

    render.output_plot(to_file)


def plot_gaussian(gaussian, title, to_file):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(np.arange(len(gaussian)), gaussian, 'g', alpha=0.7, linewidth=2, label='Beta')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Amplitude')
    ax.title.set_text(title)
    # ax.set_ylim([0, 1])
    ax.grid(True)

    render.output_plot(to_file)


def optimise_model(expected, observed, title, to_file):
    f, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(range(len(observed)), observed, 'b', alpha=0.7, linewidth=2, label='observed')
    ax.plot(range(len(expected)), expected, 'r', alpha=0.7, linewidth=2, label='expected')

    ax.title.set_text(title)

    ax.set_xlabel('Month')
    ax.set_ylabel('Parthenium Population as Percentage of Total')
    ax.grid(True)

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    render.output_plot(to_file)


class PlotData:
    pass
