# folder = "modelling/progressions/2019_2020_predictions/"
# all_predictions = np.load("{folder}first_10_predictions.npy".format(folder=folder), allow_pickle=True)
# parth_presence = all_predictions.reshape((4700840, 26))
#
# subset_size = int(0.01 * len(parth_presence))
# subset_indices = np.random.choice(range(len(parth_presence)), subset_size, replace=False)
# sub_predictions = parth_presence[subset_indices]
#
# plt.plot_date(dates, np.transpose(sub_predictions))

# sim = np.load('modelling/simulations/SEIR_output.npy', allow_pickle=True)
# progression = []
# for iteration in range(len(sim)):
#     iteration_snapshot = []
#     for row in range(len(sim[iteration])):
#         char_list = list(sim[iteration][row])
#         ints = [int(i, base=16) for i in char_list]
#         iteration_snapshot.append(ints)
#     progression.append(iteration_snapshot)
#
# np.save('modelling/simulations/SEIR_simulation.npy', progression)

sim = np.load('modelling/simulations/SEIR_simulation.npy', allow_pickle=True)

# plotter = PlotData()
# for iteration in range(len(sim)):
#     plotter.replot(sim[iteration], iteration)

colours = ['navajowhite', 'mediumseagreen', 'indianred', 'black']
cmap = ListedColormap(colours)
output_folder = './modelling/simulations/SEIR_anim'

for iteration in range(len(sim)):
    figure = plt.figure()
    axes = figure.add_subplot(111)
    im = axes.imshow(sim[iteration], vmin=0, vmax=len(colours), cmap=cmap)
    figure.set_dpi(600)
    axes.axis('off')

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(
        ['Susceptible', 'Exposed', 'Infected', 'Recovered'])

    plt.savefig('{outputFolder}/sim-{iter:03d}.png'.format(outputFolder=output_folder, iter=iteration))
    plt.close()