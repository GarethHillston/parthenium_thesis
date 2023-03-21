import numpy as np
from modelling import spread_model, initialise_grid
from modelling.plot_data import PlotData


def markov(locations, max_iterations, trans_matrix):
    plotter = PlotData()

    for i in range(max_iterations):
        print("{pc} %".format(pc=(100 * i/max_iterations)))
        plotter.replot_binary(locations, i)

        locations = spread_model.markov_basic_all(locations, trans_matrix)

    plotter.replot_binary(locations, max_iterations + 1)


def markov_test(grid_size, max_iterations, trans_matrix):
    plotter = PlotData()

    locations = np.zeros(grid_size, dtype=int)

    initialise_grid.random_start(locations, len(trans_matrix))

    for i in range(max_iterations):
        print("{pc} %".format(pc=(100 * i/max_iterations)))
        plotter.replot(locations, i)

        locations = spread_model.markov_basic(locations, trans_matrix)

    plotter.replot(locations, max_iterations + 1)


class Simulate:
    pass
