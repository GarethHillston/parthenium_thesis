import matplotlib.pyplot as plt
from matplotlib import dates
import matplotlib as mpl
import matplotlib.ticker as mtick
import numpy as np
import os
from datetime import datetime
import seaborn as sns
import pandas as pd

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

band_names = dict(B02='blue', B03='green', B04='red', B05='low IR', B06='mid NIR', B07='high NIR', B08='wide NIR',
                  B8A='higher NIR', B11='1610 SWIR', B12='2190 SWIR')

eight_colours = ['black', 'red', 'grey', 'blue', 'orange', 'green', 'purple', 'pink']
month_colours = ['blue', 'green', 'red', 'purple', 'orange']
binary_colours = ['white', 'mediumseagreen']

year_bounds = [dates.datestr2num('2018-12-01'),
               dates.datestr2num('2019-12-01'),
               dates.datestr2num('2020-12-01'),
               dates.datestr2num('2021-12-01'),
               dates.datestr2num('2022-12-01')]

years = ['2018', '2019', '2020', '2021', '2022']

now = datetime.now().strftime("%d_%m_%y__%H%M%S")
output_folder = "{dateTime}".format(dateTime=now)


def histogram(bands_data):
    flat_arrays = []

    for band in bands_data.keys():
        flat_arrays.append(bands_data[band].flatten())

    figure = plt.figure()
    axes2 = figure.add_subplot(111)
    axes2.hist(flat_arrays, bins=50, histtype='bar')
    plt.show()


def multi_plot(image_data, colour_map, to_file):
    cmap = plt.get_cmap(colour_map) if colour_map == str else plt.get_cmap('inferno')
    num_images = 0
    figure = plt.figure()
    # figure.set_dpi(600)

    for key in image_data.keys():
        num_images += 1

        axes = figure.add_subplot(1, len(image_data), num_images)
        axes.title.set_text(key)
        axes.axis('off')
        axes.imshow(image_data[key], cmap=cmap)

    figure.tight_layout()

    if to_file != '':
        if '.png' in to_file:
            plt.savefig(to_file)
        else:
            time = datetime.now().strftime("%H%M%S")
            plt.savefig('{outputFolder}/plot_{time}.png'.format(outputFolder=to_file, time=time))
        plt.close()
    else:
        plt.show()


def full_year_plot(image_data, to_file):
    cmap = plt.get_cmap('inferno')
    figure = plt.figure(figsize=(15, 10))
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    for i in range(12):
        month = image_data[i]
        num_images_this_month = 0

        for day in month:
            num_images_this_month += 1

            axes = figure.add_subplot(12, 7, (i * 7) + num_images_this_month)
            axes.axis('off')
            im = axes.imshow(month[day], cmap=cmap)

    plt.subplots_adjust(left=0.125,
                        right=0.4,
                        bottom=0.1,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.2)

    cbar_ax = figure.add_axes([0.45, 0.15, 0.02, 0.7])
    figure.colorbar(im, cax=cbar_ax)

    if to_file != '':
        if '.png' in to_file:
            plt.savefig(to_file)
        else:
            time = datetime.now().strftime("%H%M%S")
            plt.savefig('{outputFolder}/plot_{time}.png'.format(outputFolder=to_file, time=time))
        plt.close()
    else:
        plt.show()


def full_month_plot(image_data, to_file):
    cmap = plt.get_cmap('inferno')
    figure = plt.figure(figsize=(19.2, 10.8))

    num_images_this_month = 0

    for day in image_data:
        num_images_this_month += 1

        axes = figure.add_subplot(1, 4, num_images_this_month)
        axes.axis('off')
        im = axes.imshow(image_data[day], cmap=cmap)

    plt.subplots_adjust(left=0.125,
                        right=0.4,
                        bottom=0.1,
                        top=0.9,
                        wspace=0.2,
                        hspace=0.2)

    cbar_ax = figure.add_axes([0.45, 0.15, 0.02, 0.7])
    figure.colorbar(im, cax=cbar_ax)

    if to_file != '':
        if '.png' in to_file:
            plt.savefig(to_file)
        else:
            time = datetime.now().strftime("%H%M%S")
            plt.savefig('{outputFolder}/plot_{time}.png'.format(outputFolder=to_file, time=time))
        plt.close()
    else:
        plt.show()


def rgb_series(image_series):
    num_images = 0
    figure = plt.figure()

    for date in image_series.keys():
        num_images += 1
        rgb_cube = image_series[date]

        norm_red = normalise(rgb_cube[0])
        norm_green = normalise(rgb_cube[1])
        norm_blue = normalise(rgb_cube[2])

        image = np.dstack((norm_red, norm_green, norm_blue))

        axes = figure.add_subplot(1, len(image_series), num_images)
        axes.title.set_text(date.split('T')[0])
        axes.axis('off')
        axes.imshow(image)

    figure.tight_layout()
    plt.show()


def binary_plot(image_data, title, to_file):
    cmap = ListedColormap(binary_colours)

    figure = plt.figure(figsize=(19.2, 10.8))
    axes = figure.add_subplot(111)
    axes.imshow(image_data, cmap=cmap)
    axes.axis('off')

    if title != '':
        axes.title.set_text(title)

    output_plot(to_file)


def single_plot(image_data, title, colour_map, to_file):
    cmap = plt.get_cmap('inferno') if colour_map == str else ListedColormap(colour_map)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    im = axes.imshow(image_data)
    # figure.set_dpi(600)
    axes.axis('off')

    # values = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    # bounds = range(len(values))
    # divider = make_axes_locatable(axes)
    # cax = divider.append_axes("right", size="5%", pad=0.2)
    # cbar = plt.colorbar(im, cax=cax, cmap=cmap, spacing='proportional', boundaries=bounds)
    # cbar.set_ticks(values)
    # cbar.set_ticklabels(values)
    # cbar.ax.tick_params(labelsize=8)

    if title != '':
        axes.title.set_text(title)

    if to_file != '':
        if '.png' in to_file:
            plt.savefig(to_file)
        else:
            time = datetime.now().strftime("%H%M%S")
            plt.savefig('{outputFolder}/plot_{time}.png'.format(outputFolder=to_file, time=time))
        plt.close()
    else:
        plt.show()


def lulc_plot(image_data, title, to_file):
    cmap = ListedColormap(eight_colours)

    figure = plt.figure(figsize=(19.2, 10.2))
    axes = figure.add_subplot(111)
    im = axes.imshow(image_data, cmap=cmap)
    axes.axis('off')
    plt.colorbar(im, ticks=np.arange(0, 8))

    if title != '':
        axes.title.set_text(title)

    output_plot(to_file)


def rgb_lulc_comparison(rgb, lulc, date, to_file):
    figure = plt.figure(figsize=(10.8, 19.2))
    axes = figure.add_subplot(121)
    axes.imshow(rgb)
    axes.axis('off')
    axes.title.set_text("True colour " + date)

    cmap = ListedColormap(eight_colours)
    axes2 = figure.add_subplot(122)
    im = axes2.imshow(lulc, cmap=cmap)
    axes2.axis('off')
    axes2.title.set_text("Land classifications " + date)
    plt.colorbar(im, ticks=np.arange(0, 8))

    output_plot(to_file)


def lulc_stats_plot(stats, title, to_file):
    fig = plt.figure(figsize=(19.2, 10.8))
    ax = fig.add_subplot(111)

    percentages = np.row_stack((stats['pc0'],
                                stats['pc1'],
                                stats['pc2'],
                                stats['pc3'],
                                stats['pc4'],
                                stats['pc5'],
                                stats['pc6'],
                                stats['pc7']))

    ax.stackplot(stats['dateTime'], percentages, labels=range(8), colors=eight_colours)
    ax.legend(loc='upper left')
    ax.set_ylabel('Percent (%)')
    ax.set_xlabel('Date')
    ax.margins(0, 0)

    if title != '':
        ax.title.set_text(title)

    output_plot(to_file)


def lulc_stats_stack_plot(stats, title, to_file):
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10.8, 19.2), sharex=True, sharey=True)

    for ax, iter in zip(axs.ravel(), range(4)):
        date_set_condition = (stats['dateTime'] > year_bounds[iter]) & (stats['dateTime'] < year_bounds[iter + 1])
        bounded_subset = stats[date_set_condition]
        date_range = bounded_subset["dateTime"] - iter * 365
        percentages = np.row_stack((bounded_subset['pc0'],
                                    bounded_subset['pc1'],
                                    bounded_subset['pc2'],
                                    bounded_subset['pc3'],
                                    bounded_subset['pc4'],
                                    bounded_subset['pc5'],
                                    bounded_subset['pc6'],
                                    bounded_subset['pc7']))

        ax.stackplot(date_range, percentages, labels=range(8), colors=eight_colours)
        ax.title.set_text(years[iter] + ' / ' + years[iter + 1])
        ax.margins(0, 0)

        ymin = 0
        ymax = 1
        xmin = np.datetime64('2018-12-01')
        xmax = np.datetime64('2019-12-01')

        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        ax.vlines(x=np.datetime64('2019-01-01'),
                  ymin=ymin,
                  ymax=ymax,
                  colors='white')

    ax.legend(loc='lower right')

    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, None, ''))

    fig.text(0.5, 0.04, "Date", ha='center', fontsize=20)
    fig.text(0.04, 0.5, "Percent (%)", va='center', rotation='vertical', fontsize=20)

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def rgb_plot(rgb_data, title, to_file):
    figure = plt.figure(figsize=(19.2, 10.8))
    axes = figure.add_subplot(111)
    axes.imshow(rgb_data)
    axes.axis('off')
    plt.suptitle(title)
    output_plot(to_file)


def rgb_series_to_file(image_series):
    iteration = 0
    os.mkdir(output_folder)

    for date in image_series.keys():
        iteration += 1
        rgb_cube = image_series[date]

        norm_red = normalise(rgb_cube[0])
        norm_green = normalise(rgb_cube[1])
        norm_blue = normalise(rgb_cube[2])

        image = np.dstack((norm_red, norm_green, norm_blue))

        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.imshow(image)
        axes.axis('off')
        plt.savefig('{outputFolder}/sim-{iter:03d}.png'.format(outputFolder=output_folder, iter=iteration))
        plt.close()


def ndvi_parth_scatter(parth_per_ndvi):
    figure = plt.figure(figsize=(8, 8))
    axes = figure.add_subplot(111)
    axes.scatter(x=parth_per_ndvi["ndvi"],
                 y=parth_per_ndvi["parth"],
                 alpha=0.001, s=10)
    axes.title.set_text("Parthenium likelihood by NVDI score")

    plt.show()
    # plt.savefig('{fp}/plots/parth_vs_ndvi.png'.format(fp=fp))


def plot_stats(stats, title, vmin, vmax, main_avg, small_err, fill_err, to_file):
    other_avg = 'mean'
    if main_avg == 'mean':
        other_avg = 'median'

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    error_bars = stats["std"]
    if small_err:
        error_bars *= 0.5

    if fill_err:
        top = stats[main_avg] + stats["std"]
        bottom = stats[main_avg] - stats["std"]
        ax.fill_between(stats["dateTime"], top, bottom, alpha=.5, linewidth=0)
        ax.plot(stats["dateTime"], stats[main_avg], label=main_avg, color="indianred")
    else:
        ax.errorbar(stats["dateTime"], stats[main_avg], yerr=error_bars, label=main_avg, color="brown",
                    ecolor="indianred")

    ax.plot(stats["dateTime"], stats[other_avg], label=other_avg, color="darkslateblue")

    deliniate_year_firsts(vmax, vmin)

    add_nice_year_month_labels()

    ax.set_xlabel("date", fontsize=20)
    ax.set_ylabel("mean/median value", fontsize=20)
    ax.legend(loc='best', prop={'size': 22})
    plt.suptitle(title, fontsize=20)

    if to_file != '':
        if '.png' in to_file or '.pdf' in to_file:
            plt.savefig(to_file)
        else:
            time = datetime.now().strftime("%H%M%S")
            plt.savefig('{outputFolder}/plot_{time}.png'.format(outputFolder=to_file, time=time), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def lulc_ndvi_stats(stats, title, vmin, vmax, avg, to_file):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    for land_type in range(8):
        bounded_subset = stats.loc[stats['land class'] == land_type]
        data = bounded_subset[avg + ' ndvi']
        date_range = bounded_subset["dateTime"]
        label = 'Class ' + str(land_type)

        ax.plot(date_range, data, label=label, color=eight_colours[land_type])

    deliniate_year_firsts(vmax, vmin)

    add_nice_year_month_labels()

    ax.set_ylabel("{avg} value".format(avg=avg), fontsize=20)
    ax.set_xlabel("date", fontsize=20)
    ax.legend(loc='upper right')

    if title != '':
        ax.title.set_text(title)

    output_plot(to_file)


def lulc_signatures(stats, title, avg, to_file):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    wavelengths = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]

    for land_type in range(8):
        bounded_subset = stats.loc[stats['land class'] == land_type]
        data = bounded_subset[avg + ' ref']
        label = 'Class ' + str(land_type)
        wavelengths = bounded_subset["wavelength"]

        top = bounded_subset[avg + ' ref'] + bounded_subset["std ref"]
        bottom = bounded_subset[avg + ' ref'] - bounded_subset["std ref"]
        ax.fill_between(wavelengths, top, bottom, alpha=.5, linewidth=0, color=eight_colours[land_type])
        ax.plot(wavelengths, data, label=label, color=eight_colours[land_type])

    plt.xticks(wavelengths, rotation=45)
    ax.set_ylabel("{avg} reflectance value".format(avg=avg), fontsize=20)
    ax.set_xlabel("Wavelength in micrometres", fontsize=20)
    ax.legend(loc='upper right')

    if title != '':
        ax.title.set_text(title)

    output_plot(to_file)


def lulc_parth_stats(stats, title, avg, to_file):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    data = []
    for i in range(8):
        data.append(stats[stats["land class"] == i][avg + " parth"])

    ax.boxplot(data)

    ax.set_ylabel("{avg} Parthenium prediction".format(avg=avg), fontsize=20)
    ax.set_xlabel("Class", fontsize=20)
    ax.legend(loc='upper right')

    if title != '':
        ax.title.set_text(title)

    output_plot(to_file)


def lulc_parth_stack_stats(stats, title, avg, to_file):
    vmin = 0
    vmax = 1
    mpl.style.use('seaborn')
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10.8, 19.2), sharex=True, sharey=True)

    for ax, iter in zip(axs.ravel(), range(4)):
        date_set_condition = (stats['dateTime'] > year_bounds[iter]) & (stats['dateTime'] < year_bounds[iter + 1])
        bounded_subset = stats[date_set_condition]

        for land_type in range(8):
            land_type_subset = bounded_subset.loc[bounded_subset['land class'] == land_type]
            data = land_type_subset[avg + ' parth']
            date_range = land_type_subset["dateTime"] - iter * 365
            label = 'Class ' + str(land_type)

            ax.plot(date_range, data, label=label, color=eight_colours[land_type])

            xmin = np.datetime64('2018-12-01')
            xmax = np.datetime64('2019-12-01')

            ax.set(xlim=(xmin, xmax), ylim=(vmin, vmax))

            ax.vlines(x=np.datetime64('2019-01-01'),
                      ymin=vmin,
                      ymax=vmax,
                      colors='grey')

    plt.legend(bbox_to_anchor=(1, 1), loc="lower right")

    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))

    fig.text(0.5, 0.04, "Date", ha='center', fontsize=20)
    fig.text(0.04, 0.5, "{avg} value".format(avg=avg), va='center', rotation='vertical', fontsize=20)
    ax.legend(loc='lower right')

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def lulc_ndvi_stack_stats(stats, title, vmin, vmax, avg, to_file):
    mpl.style.use('seaborn')
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10.8, 12), sharex=True, sharey=True)

    for ax, iter in zip(axs.ravel(), range(4)):
        date_set_condition = (stats['dateTime'] > year_bounds[iter]) & (stats['dateTime'] < year_bounds[iter + 1])
        bounded_subset = stats[date_set_condition]

        for land_type in range(8):
            land_type_subset = bounded_subset.loc[bounded_subset['land class'] == land_type]
            data = land_type_subset[avg + ' ndvi']
            date_range = land_type_subset["dateTime"] - iter * 365
            label = 'Class ' + str(land_type + 1)

            ax.plot(date_range, data, label=label, color=eight_colours[land_type])

            xmin = np.datetime64('2018-12-01')
            xmax = np.datetime64('2019-12-01')

            ax.set(xlim=(xmin, xmax), ylim=(vmin, vmax))

            ax.vlines(x=np.datetime64('2019-01-01'),
                      ymin=vmin,
                      ymax=vmax,
                      colors='grey')

        ax.set_title('{} / {}'.format((2018 + iter), (2019 + iter)), fontsize=16)

    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))

    fig.text(0.5, 0.04, "Date", ha='center', fontsize=20)
    fig.text(0.04, 0.5, "{avg} NDVI".format(avg=avg.capitalize()), va='center', rotation='vertical', fontsize=20)
    ax.legend(loc='lower right', prop={'size': 11})

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def stack_stats(stats, title, main_avg, small_err, fill_err, to_file):
    other_avg = 'mean'
    if main_avg == 'mean':
        other_avg = 'median'

    error_bars = stats["std"]
    if small_err:
        error_bars *= 0.5

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 18), sharex=True, sharey=True)

    for ax, iter in zip(axs.ravel(), range(4)):
        date_set_condition = (stats['dateTime'] > year_bounds[iter]) & (stats['dateTime'] < year_bounds[iter + 1])
        bounded_subset = stats[date_set_condition]
        main_avg_data = bounded_subset[main_avg]
        other_avg_data = bounded_subset[other_avg]
        date_range = bounded_subset["dateTime"] - iter * 365
        std = bounded_subset["std"]

        if fill_err:
            top = main_avg_data + std
            bottom = main_avg_data - std
            ax.fill_between(date_range, top, bottom, alpha=.5, linewidth=0)
            ax.plot(date_range, main_avg_data, label=main_avg, color="indianred")
        else:
            ax.errorbar(date_range, main_avg_data, yerr=error_bars, label=main_avg, color="brown", ecolor="indianred")

        ax.plot(date_range, other_avg_data, label=other_avg, color="darkslateblue")
        ax.title.set_text(years[iter] + ' - ' + years[iter + 1])

        ymin = 0
        ymax = 0.85
        xmin = np.datetime64('2018-12-01')
        xmax = np.datetime64('2019-12-01')

        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        ax.vlines(x=np.datetime64('2019-01-01'),
                  ymin=ymin,
                  ymax=ymax,
                  colors='grey')

        ax.legend(loc='upper center')
        ax.grid()

    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))

    fig.text(0.5, 0.04, "Date", ha='center', fontsize=20)
    fig.text(0.04, 0.5, "Mean/Median Value", va='center', rotation='vertical', fontsize=20)

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def binary_plot(binaries, clouds, title, to_file):
    if clouds:
        cloud_cover = pd.read_pickle('data_overview/clouds.pkl')

    mpl.style.use('seaborn')
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10.8, 19.2), sharex=True, sharey=True)

    for ax, iter in zip(axs.ravel(), range(4)):
        date_set_condition = (binaries['dateTime'] > year_bounds[iter]) & (binaries['dateTime'] < year_bounds[iter + 1])
        bounded_subset = binaries[date_set_condition]
        date_range = bounded_subset["dateTime"] - iter * 365
        data = bounded_subset["pc_parth"]

        if clouds:
            cloud_data = cloud_cover[date_set_condition]
            ax.plot(date_range, cloud_data["pc_masked"], color="cornflowerblue")
        ax.plot(date_range, data, color="indianred")
        ax.title.set_text(years[iter] + ' - ' + years[iter + 1])

        ymin = 0
        ymax = 1
        xmin = np.datetime64('2018-12-01')
        xmax = np.datetime64('2019-12-01')

        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

        ax.vlines(x=np.datetime64('2019-01-01'),
                  ymin=ymin,
                  ymax=ymax,
                  colors='grey')

        ax.legend(loc='best')

    ax.xaxis.set_major_locator(dates.MonthLocator())
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))

    fig.text(0.5, 0.04, "Date", ha='center', fontsize=20)
    fig.text(0.04, 0.5, "Parthenium Cover", va='center', rotation='vertical', fontsize=20)

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def monthly_cover(cover, title, to_file):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    for year, colour in zip(years, month_colours):
        year_data = cover[cover["year"] == int(year)]
        months = year_data["month"]
        ax.plot(months, year_data["pc_parth"], color=colour, marker='o')
        ax.plot(months, year_data["pc_parth"], color=colour, label=year)

    ax.legend(loc='lower right')

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def averaged_monthly_cover(cover, title, to_file):
    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    ax.plot(range(1, 13), cover["avg_pc_parth"], color='r')

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def temperature(temp, title, to_file):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    subset = temp.dropna()

    ax.plot(subset["dateTime"], subset["temp_mean"], color='r')

    add_nice_year_month_labels()

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def precipitation(prec, title, to_file):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    ax.plot(prec["dateTime"], prec["prec_mean"], color='b')

    add_nice_year_month_labels()

    if title != '':
        plt.suptitle(title, fontsize=20)

    output_plot(to_file)


def normalise(array):
    divisor = (np.max(array) - np.min(array)) if np.max(array) != np.min(array) else 1.0
    return (array - np.min(array)) / divisor


def add_nice_year_month_labels():
    plt.gca().xaxis.set_minor_locator(dates.MonthLocator())
    plt.gca().xaxis.set_minor_formatter(dates.DateFormatter('%b'))
    plt.setp(plt.gca().xaxis.get_minorticklabels(), rotation=45)

    plt.gca().xaxis.set_major_locator(dates.YearLocator())
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%Y'))
    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45)


def deliniate_year_firsts(vmax, vmin):
    year_firsts = (np.datetime64('2019-01-01'),
                   np.datetime64('2020-01-01'),
                   np.datetime64('2021-01-01'),
                   np.datetime64('2022-01-01'))
    plt.vlines(x=year_firsts,
               ymin=vmin,
               ymax=vmax,
               colors='grey')


def output_plot(to_file):
    if to_file != '':
        if '.png' in to_file:
            plt.savefig(to_file)
        else:
            time = datetime.now().strftime("%H%M%S")
            plt.savefig('{outputFolder}/plot_{time}.png'.format(outputFolder=to_file, time=time))
        plt.close()
    else:
        plt.show()


class Render:
    pass
