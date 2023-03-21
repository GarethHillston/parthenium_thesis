import numpy as np
from imaging import get_data, indices, classify, render


def histogram(raw_data, date, bands):
    data = {}

    for band in bands:
        data.update({band: raw_data.sel(band=band, date=date).reflectance.data})

    render.histogram(data)


def multi_plot(raw_data, date, bands):
    data = {}

    for band in bands:
        data.update({band: raw_data.sel(band=band, date=date).reflectance.data})

    render.multi_plot(data, date)


def generate_rgb_image(raw_data, date):
    red_band = get_data.by_band_and_date(raw_data, 'B04', date)
    green_band = get_data.by_band_and_date(raw_data, 'B03', date)
    blue_band = get_data.by_band_and_date(raw_data, 'B02', date)

    rgb_cube = [red_band, green_band, blue_band]

    norm_red = get_data.nan_normalise(rgb_cube[0])
    norm_green = get_data.nan_normalise(rgb_cube[1])
    norm_blue = get_data.nan_normalise(rgb_cube[2])

    rgb = np.transpose([norm_red, norm_green, norm_blue], (1, 2, 0))

    return rgb


def rgb_series(raw_data, date_range, to_file=False):
    image_series = {}

    for date in date_range:
        image_series.update(generate_rgb_image(raw_data, date))

    if to_file:
        render.rgb_series_to_file(image_series)
    else:
        render.rgb_series(image_series)


def bare_soil_index(raw_data, date):
    soil_index = indices.bare_soil_index(raw_data, date)

    nir = get_data.by_band_and_date(raw_data, 'B08', date)
    low_swir = get_data.by_band_and_date(raw_data, 'B11', date)

    norm_index = get_data.normalise(soil_index)
    norm_nir = get_data.normalise(nir)
    norm_swir = get_data.normalise(low_swir)

    image_data = np.dstack((norm_index, norm_nir, norm_swir))

    render.rgb_plot(image_data)


def classification_progression_ndvi(raw_data, dates, start_date, end_date, image_size):
    image_data = {}
    clusters = 8

    training_set = []
    for date in dates:
        training_set.append(indices.ndvi(raw_data, date))
    training_set = np.array(training_set)
    classifier = classify.train_kmeans(training_set, 1, clusters)

    start_set = indices.ndvi(raw_data, start_date)
    start_set = start_set.reshape(np.prod(image_size), 1)
    start_results = classify.run_classification(classifier, start_set, image_size)
    image_data[start_date] = start_results

    end_set = indices.ndvi(raw_data, end_date)
    end_set = end_set.reshape(np.prod(image_size), 1)
    end_results = classify.run_classification(classifier, end_set, image_size)
    image_data[end_date] = end_results

    colours = ['plum', 'coral', 'lightgreen', 'paleturquoise', 'black', 'white', 'silver', 'firebrick', 'khaki', 'royalblue', 'forestgreen']
    colour_subset = colours[0:clusters]
    render.multi_plot(image_data, colour_subset)

    return [start_results, end_results]


def cloud_demonstration():
    raw_data = np.load('{fp}/raw_data/scl/2019_1_scl.npy'.format(fp=fp), allow_pickle=True).item()

    date = '2019-05-11'

    SCL = raw_data.get(date)

    bad = np.where(SCL == 1, 0, 0)
    dark = np.where(SCL == 2, 0, 0)
    shadow = np.where(SCL == 3, 1, 0)
    veg = np.where(SCL == 4, 0, 0)
    soil = np.where(SCL == 5, 0, 0)
    water = np.where(SCL == 6, 2, 0)
    low = np.where(SCL == 7, 3, 0)
    med = np.where(SCL == 8, 4, 0)
    high = np.where(SCL == 9, 5, 0)
    cirrus = np.where(SCL == 10, 6, 0)
    ice = np.where(SCL == 11, 7, 0)

    image_data = bad + dark + shadow + veg + soil + water + low + med + high + cirrus + ice

    colours = ['white', 'royalblue', 'lightsteelblue', 'mistyrose', 'lightcoral', 'indianred', 'maroon', 'black']
    cmap = ListedColormap(colours)

    figure = plt.figure(figsize=(12, 10))
    axes = figure.add_subplot(111)
    im = axes.imshow(image_data, vmin=0, vmax=len(colours), cmap=cmap)
    axes.axis('off')
    axes.title.set_text(date)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    cbar.set_ticklabels(
        ['Surface', 'Water', 'Ice', 'Low cloud prob.', 'Medium cloud prob.', 'High cloud prob.',
         'Cirrus clouds', 'Cloud shadow'])

    # plt.savefig('./imaging/plots/coloured/scl_coloured/{date}_cloud_highlights.png'.format(date=date))
    plt.show()
    plt.close()


def mask_set():
    return
    # scl_mask = get_data.cloud_water_mask(raw_data, dates[0])
    # band_data = get_data.all_bands_by_date(raw_data, dates[0])
    # masked_set = []
    #
    # for i in range(image_size[0]):
    #     for j in range(image_size[1]):
    #         band_data[i][j] = band_data[i][j] * scl_mask[i][j]
    #         if np.sum(band_data[i][j]) != 0:
    #             masked_set.append(band_data[i][j])
    #
    # masked_set = np.array(masked_set)
    # num_classes = 8
    # classifier = classify.train_kmeans(masked_set, 10, num_classes)
    #
    # pickle.dump(classifier, open("./imaging/classifiers/allBandOneDateNoCloud.pkl", "wb"))


def class_parth_stats():
    predictions = np.load('./progressions/parth_presence.npy')
    predictions = np.transpose(predictions)

    classifications = np.load('./progressions/classification.npy')

    class_stats = [[], [], [], [], [], [], [], []]

    for predict_row, class_row in zip(predictions, classifications):
        for parth, cluster in zip(predict_row, class_row):
            class_stats[cluster].append(parth)

    class_stat_size = len(class_stats)
    for class_row, number in zip(class_stats, range(class_stat_size)):
        print('Class ', number)
        print('Mean   - ', np.nanmean(class_row))
        print('Median - ', np.nanmedian(class_row))
        print('STD    - ', np.nanstd(class_row))
        print('Total  - ', len(class_row))
        print()
    total = 0
    for class_row in class_stats:
        total += np.nansum(class_row)

    proportions = np.empty(class_stat_size)

    for class_row, i in zip(class_stats, range(class_stat_size)):
        proportions[i] = np.nansum(class_row) / total

    print(proportions)
    print(np.sum(proportions))

    # figure = plt.figure()
    #
    # axes = figure.add_subplot(111)
    # im = axes.imshow(output, cmap=plt.get_cmap('inferno'))
    # axes.axis('off')
    # axes.title.set_text("Predicted Parthenium")
    #
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # divider = make_axes_locatable(axes)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im, cax=cax)
    #
    # # plt.savefig('imaging/plots/comparison.png')
    # plt.show()


def Parth_SCL_visualisation():
    for i in range(len(date_times)):
        date = date_times[i]
        date_neat = date.split('T')[0]

        binary_prediction = np.load('./progressions/bad_19_20_binaries_nanned/bin_predict_' + date_neat + '.npy')
        SCL = raw_data.sel(date=date).variables['SCL']

        """
                     0 - absence  - brown
                     1 - presence - green
            SCL
            1/2  -   2 - no good  - black
            3    -   3 - shadows  - dark grey
            6    -   4 - water    - blue
            8-10 -   5 - clouds   - grey
            11   -   6 - snow     - white
        """

        bad = np.where(SCL == 1, 2, 0)
        dark = np.where(SCL == 2, 2, 0)
        shadow = np.where(SCL == 3, 3, 0)
        veg = np.where(SCL == 4, 1, 0)
        soil = np.where(SCL == 5, 1, 0)
        water = np.where(SCL == 6, 4, 0)
        low = np.where(SCL == 7, 1, 0)
        med = np.where(SCL == 8, 5, 0)
        high = np.where(SCL == 9, 5, 0)
        cirrus = np.where(SCL == 10, 5, 0)
        ice = np.where(SCL == 11, 6, 0)

        mask = bad + dark + shadow + veg + soil + water + low + med + high + cirrus + ice

        image_data = np.where(mask == 1, binary_prediction, mask)

        colours = ['peachpuff', 'mediumseagreen', 'black', 'dimgrey', 'cornflowerblue', 'lightgrey', 'white']
        cmap = ListedColormap(colours)

        figure = plt.figure()
        axes = figure.add_subplot(111)
        im = axes.imshow(image_data, vmin=0, vmax=len(colours), cmap=cmap)
        figure.set_dpi(600)
        axes.axis('off')
        axes.title.set_text(date_neat)

        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        cbar.set_ticklabels(
            ['No Parthenium', 'Parthenium', 'Dark/defective', 'Cloud shadow', 'Water', 'Cloud', 'Ice/snow'])

        plt.savefig('./imaging/plots/coloured/' + date_neat + '.png')
        plt.close()


def SCL_visualistaion():
    raw_data = np.load('{fp}/raw_data/scl/2019_1_scl.npy'.format(fp=fp), allow_pickle=True).item()

    # for i in range(len(date_times)):
    date = '2019-04-06'

    SCL = raw_data.get(date)

    bad = np.where(SCL == 1, 2, 0)
    dark = np.where(SCL == 2, 2, 0)
    shadow = np.where(SCL == 3, 3, 0)
    veg = np.where(SCL == 4, 1, 0)
    soil = np.where(SCL == 5, 0, 0)
    water = np.where(SCL == 6, 4, 0)
    low = np.where(SCL == 7, 0, 0)
    med = np.where(SCL == 8, 5, 0)
    high = np.where(SCL == 9, 5, 0)
    cirrus = np.where(SCL == 10, 5, 0)
    ice = np.where(SCL == 11, 6, 0)

    image_data = bad + dark + shadow + veg + soil + water + low + med + high + cirrus + ice

    colours = ['peachpuff', 'mediumseagreen', 'black', 'dimgrey', 'cornflowerblue', 'lightgrey', 'white']
    cmap = ListedColormap(colours)

    figure = plt.figure(figsize=(19.2, 10.8))
    axes = figure.add_subplot(111)
    im = axes.imshow(image_data, vmin=0, vmax=len(colours), cmap=cmap)
    axes.axis('off')
    axes.title.set_text(date)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    cbar.set_ticklabels(
        ['Bare soil', 'Vegetation', 'Dark/defective', 'Cloud shadow', 'Water', 'Cloud', 'Ice/snow'])

    plt.savefig('./imaging/plots/coloured/scl_coloured/' + date + '.png')
    plt.show()
    plt.close()


def SCL_visualistaion():
    raw_data = np.load('{fp}/raw_data/scl/2019_1_scl.npy'.format(fp=fp), allow_pickle=True).item()

    # for i in range(len(date_times)):
    date = '2019-04-06'

    SCL = raw_data.get(date)

    bad = np.where(SCL == 1, 0, 0)
    dark = np.where(SCL == 2, 0, 0)
    shadow = np.where(SCL == 3, 1, 0)
    veg = np.where(SCL == 4, 0, 0)
    soil = np.where(SCL == 5, 0, 0)
    water = np.where(SCL == 6, 2, 0)
    low = np.where(SCL == 7, 3, 0)
    med = np.where(SCL == 8, 4, 0)
    high = np.where(SCL == 9, 5, 0)
    cirrus = np.where(SCL == 10, 6, 0)
    ice = np.where(SCL == 11, 7, 0)

    image_data = bad + dark + shadow + veg + soil + water + low + med + high + cirrus + ice

    colours = ['white', 'black', 'royalblue', 'mistyrose', 'lightcoral', 'indianred', 'maroon', 'lightsteelblue']
    cmap = ListedColormap(colours)

    figure = plt.figure(figsize=(19.2, 10.8))
    axes = figure.add_subplot(111)
    im = axes.imshow(image_data, vmin=0, vmax=len(colours), cmap=cmap)
    axes.axis('off')
    axes.title.set_text(date)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    cbar.set_ticklabels(
        ['Surface', 'Cloud shadow', 'Water', 'Low cloud prob.', 'Medium cloud prob.', 'High cloud prob.',
         'Cirrus clouds', 'Ice'])

    plt.savefig('./imaging/plots/coloured/scl_coloured/{date}_cloud_highlights.png')
    plt.show()
    plt.close()


class Display:
    pass
