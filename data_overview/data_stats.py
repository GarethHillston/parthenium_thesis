import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import pandas as pd
from matplotlib import dates
import pickle

from imaging import render, indices, get_data

fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'


def get_ndvi_stats():
    pass


def plot_yearly_ndvi():
    year = '2019'
    ndvi = np.load('{fp}/ndvi/{year}_ndvi.npy'.format(fp=fp, year=year), allow_pickle=True).item()

    year_cube = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

    for date in ndvi.keys():
        month = int(date.split('-')[1]) - 1
        year_cube[month][date] = ndvi[date]

    render.full_year_plot(year_cube, '{fp}/ndvi/plots/{year}.png'.format(fp=fp, year=year))
    # render.full_year_plot(year_cube, '')


def plot_month_ndvi():
    year = '2022'
    ndvi = np.load('{fp}/ndvi/{year}_ndvi.npy'.format(fp=fp, year=year), allow_pickle=True).item()

    dates = {}

    for date in ndvi.keys():
        dates[date] = ndvi[date]

    render.full_month_plot(dates, '{fp}/ndvi/plots/{year}.png'.format(fp=fp, year=year))
    # render.full_month_plot(dates, '')

def convert_data_format():
    ndvi = np.load('{fp}/ndvi/2019_ndvi.npy'.format(fp=fp), allow_pickle=True).item()

    print(ndvi.keys())

    out_ndvi = {}

    for date in ndvi.keys():
        if date != '2019-10-03':
            out_ndvi[date] = ndvi[date]

    print(out_ndvi.keys())

    np.save('{}/ndvi/2019_ndvi.npy'.format(fp), out_ndvi)


def mask_date_range():
    scl = np.load('{fp}/raw_data/scl/December_2018-2021.npy'.format(fp=fp), allow_pickle=True).item()
    ref = np.load('{fp}/raw_data/reflectance/December_2018-2021.npy'.format(fp=fp), allow_pickle=True).item()

    masked_output = get_data.apply_light_mask(ref, scl)

    np.save('{fp}/raw_data/mask_ref/Decembers_light_mask.npy'.format(fp=fp), masked_output)


def plot_multi_months():
    ndvi = np.load('{fp}/ndvi/Decembers_ndvi.npy'.format(fp=fp), allow_pickle=True).item()

    flat_array = [[], [], [], []]

    for date in ndvi.keys():
        index = int(date.split('-')[0]) - 2018
        flat_array[index].append(ndvi[date])

    cmap = plt.get_cmap('inferno')
    figure = plt.figure()
    figure.set_dpi(600)

    for i in range(4):
        month = flat_array[i]
        num_images_this_month = 0

        for j in range(len(month)):
            num_images_this_month += 1

            axes = figure.add_subplot(4, 7, (i * 7) + num_images_this_month)
            axes.axis('off')
            axes.imshow(month[j], cmap=cmap)

    plt.savefig('{fp}/ndvi/plots/decembers.png'.format(fp=fp))
    plt.show()


def plot_multi_months_RGB():
    ref = np.load('{fp}/raw_data/mask_ref/Decembers_masked.npy'.format(fp=fp), allow_pickle=True).item()

    flat_array = [[], [], [], []]
    num_years = len(flat_array)
    max_days_per_month = 7

    for date in ref.keys():
        index = get_data.year(date) - 2018
        rgb_stack = get_data.generate_RGB(ref[date])
        flat_array[index].append(rgb_stack)

    figure = plt.figure()
    figure.set_dpi(600)

    for i in range(num_years):
        month = flat_array[i]
        num_images_this_month = 0

        for j in range(len(month)):
            num_images_this_month += 1

            axes = figure.add_subplot(
                num_years,
                max_days_per_month,
                (i * max_days_per_month) + num_images_this_month)
            axes.axis('off')
            axes.imshow(month[j])

    plt.savefig('{fp}/plots/decembers_RGB.png'.format(fp=fp))
    plt.show()


def plot_dec_vs_oct():
    ndvi_dec = np.load('{fp}/ndvi/Decembers_ndvi.npy'.format(fp=fp), allow_pickle=True).item()
    ndvi_oct = np.load('{fp}/ndvi/Octobers_ndvi.npy'.format(fp=fp), allow_pickle=True).item()

    year_set = {'Oct': {}, 'Dec': {}}
    max_num_days = 7
    year = 2021

    for date in ndvi_dec.keys():
        if get_data.year(date) == year:
            year_set['Dec'][date] = ndvi_dec[date]

    for date in ndvi_oct.keys():
        if get_data.year(date) == year:
            year_set['Oct'][date] = ndvi_oct[date]

    figure, axs = plt.subplots(nrows=2, ncols=max_num_days, sharex=True, sharey=True)
    figure.suptitle('October vs December NDVI {}'.format(year))
    figure.set_dpi(600)
    plt.xlim(-1, 1)

    img_index = 0
    for date in year_set['Oct'].keys():
        ax = axs[0][img_index]
        ax.hist(year_set['Oct'][date].flatten(), bins=50, histtype='bar')
        ax.set_title(date, fontsize=9)
        img_index += 1

    img_index = 0
    for date in year_set['Dec'].keys():
        ax = axs[1][img_index]
        ax.hist(year_set['Dec'][date].flatten(), bins=50, histtype='bar')
        ax.set_title(date, fontsize=9)
        img_index += 1

    plt.savefig('{fp}/ndvi/plots/dec_oct_histograms_{year}.png'.format(fp=fp, year=year))
    plt.show()


def def_generate_ndvi_stats():
    ndvi_dec = np.load('{fp}/ndvi/Decembers_ndvi.npy'.format(fp=fp), allow_pickle=True).item()
    ndvi_oct = np.load('{fp}/ndvi/Octobers_ndvi.npy'.format(fp=fp), allow_pickle=True).item()

    oct_dec_per_year = {}

    for year in ['2019', '2020', '2021']:
        year_set = {'Oct': {}, 'Dec': {}}

        for date in ndvi_dec.keys():
            if date.split('-')[0] == year:
                year_set['Dec'][date] = ndvi_dec[date]

        for date in ndvi_oct.keys():
            if date.split('-')[0] == year:
                year_set['Oct'][date] = ndvi_oct[date]

        oct_dec_per_year[year] = year_set

    medians = {}
    stds = {}

    for year in ['2019', '2020', '2021']:
        medians_year = {'Oct': {}, 'Dec': {}}
        stds_year = {'Oct': {}, 'Dec': {}}

        for month in medians_year.keys():
            for date in oct_dec_per_year[year][month]:
                medians_year[month][date] = np.nanmedian(oct_dec_per_year[year][month][date])
                stds_year[month][date] = np.nanstd(oct_dec_per_year[year][month][date])

        medians[year] = medians_year
        stds[year] = stds_year

    print(medians)
    print(stds)

    np.save('{fp}/ndvi/oct_dec_stats.npy'.format(fp=fp), {'medians': medians, 'stds': stds})


def generate_ndvi_stats():
    fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'
    ndvi = np.load('{fp}/ndvi/2018_ndvi.npy'.format(fp=fp), allow_pickle=True).item()
    ndvi.update(np.load('{fp}/ndvi/2019_ndvi.npy'.format(fp=fp), allow_pickle=True).item())
    ndvi.update(np.load('{fp}/ndvi/2020_ndvi.npy'.format(fp=fp), allow_pickle=True).item())
    ndvi.update(np.load('{fp}/ndvi/2021_ndvi.npy'.format(fp=fp), allow_pickle=True).item())
    ndvi.update(np.load('{fp}/ndvi/2022_ndvi.npy'.format(fp=fp), allow_pickle=True).item())

    stats_array = [[], [], [], [], [], [], []]

    for date in ndvi.keys():
        stats_array[0].append(get_data.year(date))
        stats_array[1].append(get_data.month(date))
        stats_array[2].append(get_data.day(date))
        stats_array[4].append(np.nanmedian(ndvi[date]))
        stats_array[5].append(np.nanmean(ndvi[date]))
        stats_array[6].append(np.nanstd(ndvi[date]))

    stats_array[3] = Time(list(ndvi.keys()), format='isot', scale='utc')

    stats = pd.DataFrame({
        'year': stats_array[0],
        'month': stats_array[1],
        'day': stats_array[2],
        'dateTime': stats_array[3],
        'median': stats_array[4],
        'mean': stats_array[5],
        'std': stats_array[6],
    })

    print(stats)
    stats.to_pickle('{fp}/ndvi/ndvi_stats.pkl'.format(fp=fp))


def generate_lulc_spectral_signatures(ref, lulc, file_name):
    ref = np.transpose(ref, (1, 2, 0))
    shape = np.shape(ref)
    flattened_image = np.prod(shape[:-1])
    ref = ref.reshape(flattened_image, 10)

    lulc = lulc.flatten()

    class_refs = [[], [], [], [], [], [], [], []]

    for land, surface in zip(lulc, ref):
        if land > -1:
            class_refs[land].append(surface)

    class_ref_stats = {
        "band": [],
        "wavelength": [],
        "land class": [],
        "median ref": [],
        "mean ref": [],
        "std ref": []
    }

    band_names = ['B02 - blue', 'B03 - green', 'B04 - red', 'B05 - low IR', 'B06 - mid NIR',
                  'B07 - high NIR', 'B08 - wide NIR', 'B8A - higher NIR', 'B11 - 1610 SWIR',
                  'B12 - 2190 SWIR']

    wavelengths = [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]

    for land_type in range(8):
        band_numbers = np.transpose(class_refs[land_type])
        for band in range(len(band_numbers)):
            class_ref_stats["band"].append(band_names[band])
            class_ref_stats["wavelength"].append(wavelengths[band])
            class_ref_stats["land class"].append(land_type)
            class_ref_stats["median ref"].append(np.nanmedian(band_numbers[band]))
            class_ref_stats["mean ref"].append(np.nanmean(band_numbers[band]))
            class_ref_stats["std ref"].append(np.nanstd(band_numbers[band]))

    stats_frame = pd.DataFrame(class_ref_stats)
    print(stats_frame)

    stats_frame.to_pickle(file_name)
    print("Saved to {file}".format(file=file_name))


def predict_lulc():
    ref = np.load('{fp}/raw_data/mask_ref/2019_1_masked.npy'.format(fp=fp), allow_pickle=True).item()
    ref.update(np.load('{fp}/raw_data/mask_ref/2019_2_masked.npy'.format(fp=fp), allow_pickle=True).item())
    scl = np.load('{fp}/raw_data/scl/2019_1_scl.npy'.format(fp=fp), allow_pickle=True).item()
    scl.update(np.load('{fp}/raw_data/scl/2019_2_scl.npy'.format(fp=fp), allow_pickle=True).item())

    # dates = ['2019-01-06', '2019-02-25', '2019-03-22', '2019-04-21', '2019-05-26', '2019-06-20', '2019-07-15', '2019-08-19', '2019-09-18', '2019-10-23', '2019-11-17', '2019-12-07']
    #
    # ref_array = []
    # for date in dates:
    #     ref_array.append(np.transpose(ref[date], (1, 2, 0)))
    #
    # training_data = np.array(ref_array)
    #
    # feature_count = 10
    # class_count = 8
    # kmeans = classify.train_kmeans(training_data, feature_count, class_count, 'remove')
    # pickle.dump(kmeans, open('{fp}/lulc/lulc_model_19.pkl'.format(fp=fp), 'wb'))

    cmap = render.eight_colours

    test_data = {}
    for date in ref.keys():
        test_data[date] = np.transpose(ref[date], (1, 2, 0))

    kmeans = pickle.load(open('{fp}/lulc/lulc_model_19.pkl'.format(fp=fp), 'rb'))
    results = {}

    for date in test_data.keys():
        classified_data = classify.run_classification(kmeans, test_data[date], 'zero')
        results[date] = get_data.apply_standard_mask(classified_data, scl[date], -1)
        # render.single_plot(classification, date, cmap, '{fp}/lulc/plots/{date}_lulc.png'.format(fp=fp, date=date))

    # save output
    np.save('{fp}/lulc/2019_lulc_predictions.npy'.format(fp=fp), results)


def convert_date_time(stats):
    print(stats)

    new_dates = []
    for date in stats["dateTime"]:
        new_date = dates.datestr2num(date)
        new_dates.append(new_date)

    print(new_dates)

    stats["dateTime"] = new_dates

    print(stats)

    return stats


def calculate_lulc_ndvi_stats():
    lulc = np.load('{fp}/lulc/all_lulc_predictions.npy'.format(fp=fp), allow_pickle=True).item()
    ndvi = np.load('{fp}/ndvi/all_ndvis.npy'.format(fp=fp), allow_pickle=True).item()

    ndvi_lulc_stats = {'dateTime': [],
                       'land class': [],
                       'mean ndvi': [],
                       'median ndvi': [],
                       'std ndvi': [],
                       'ndvi count': []}

    for date in lulc.keys():
        ndvis_per_class = [[], [], [], [], [], [], [], []]
        flat_lulc = lulc[date].flatten()
        flat_ndvi = ndvi[date].flatten()

        for lulc_class, ndvi_score in zip(flat_lulc, flat_ndvi):
            if lulc_class > -1:
                ndvis_per_class[lulc_class].append(ndvi_score)

        for land_type in range(8):
            ndvis_for_this_type = ndvis_per_class[land_type]
            ndvi_count = len(ndvis_for_this_type)
            mean = 0
            median = 0
            std = 0
            if ndvi_count > 0:
                mean = np.nanmean(ndvis_for_this_type)
                median = np.nanmedian(ndvis_for_this_type)
                std = np.nanstd(ndvis_for_this_type)

            ndvi_lulc_stats['dateTime'].append(dates.datestr2num(date))
            ndvi_lulc_stats['land class'].append(land_type)
            ndvi_lulc_stats['mean ndvi'].append(mean)
            ndvi_lulc_stats['median ndvi'].append(median)
            ndvi_lulc_stats['std ndvi'].append(std)
            ndvi_lulc_stats['ndvi count'].append(ndvi_count)

    np.save('{fp}/ndvi_vs_lulc_dict.npy'.format(fp=fp), ndvi_lulc_stats)

    ndvi_vs_lulc = pd.DataFrame(ndvi_lulc_stats)

    ndvi_vs_lulc.to_pickle('{fp}/ndvi_vs_lulc.pkl'.format(fp=fp))

    print(ndvi_vs_lulc)


def calculate_ndvi_parth_stats():
    ndvi = np.load('{fp}/ndvi/all_ndvis.npy'.format(fp=fp), allow_pickle=True).item()
    parth = np.load('{fp}/parth_presence/all_predictions.npy'.format(fp=fp), allow_pickle=True).item()

    parth_per_ndvi = {"dateTime": [],
                      "ndvi": [],
                      "parth": []}

    date = '2018-12-27'
    # for date in ndvi.keys():
    flat_ndvi = ndvi[date].flatten()
    flat_parth = parth[date].flatten()

    for i in range(len(flat_ndvi)):
        if flat_ndvi[i] and flat_parth[i]:
            parth_per_ndvi['dateTime'].append(dates.datestr2num(date))
            parth_per_ndvi['ndvi'].append(flat_ndvi[i])
            parth_per_ndvi['parth'].append(flat_parth[i])

    parth_per_ndvi_frame = pd.DataFrame(parth_per_ndvi)

    print(parth_per_ndvi_frame)

    parth_per_ndvi_frame.to_pickle('{fp}/parth_per_ndvi_2018-12-27.pkl'.format(fp=fp))


def calculate_parth_lulc_stats():
    fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'
    parthenium = np.load('{fp}/parth_presence/all_predictions.npy'.format(fp=fp), allow_pickle=True).item()
    land_classifications = np.load('{fp}/lulc/all_lulc_predictions.npy'.format(fp=fp), allow_pickle=True).item()

    parth_lulc_stats = {
        "land class": [],
        "date": [],
        "median parth": [],
        "mean parth": [],
        "std parth": [],
        "data count": []
    }

    for date in parthenium.keys():
        parth = parthenium[date]
        lulc = land_classifications[date]
        lulc_parth_scores = [[], [], [], [], [], [], [], []]

        for p_obs, l_obs in zip(parth.flatten(), lulc.flatten()):
            if l_obs > -1:
                lulc_parth_scores[l_obs].append(p_obs)

        for land_class in range(8):
            parth_lulc_stats["date"].append(date)
            parth_lulc_stats["land class"].append(land_class)

            scores_this_class = lulc_parth_scores[land_class]
            non_nan_sum = np.sum(~np.isnan(scores_this_class))

            if non_nan_sum > 0:
                parth_lulc_stats["median parth"].append(np.nanmedian(scores_this_class))
                parth_lulc_stats["mean parth"].append(np.nanmean(scores_this_class))
                parth_lulc_stats["std parth"].append(np.nanstd(scores_this_class))
                parth_lulc_stats["data count"].append(non_nan_sum)
            else:
                parth_lulc_stats["median parth"].append(0)
                parth_lulc_stats["mean parth"].append(0)
                parth_lulc_stats["std parth"].append(0)
                parth_lulc_stats["data count"].append(0)

    parth_lulc_frame = pd.DataFrame(parth_lulc_stats)

    print(parth_lulc_frame)

    parth_lulc_frame.to_pickle('{fp}/parth_lulc_stats.pkl'.format(fp=fp))


def calculate_parth_lulc_stats_single_date():
    date = '2019-06-20'
    # parthenium = np.load('{fp}/parth_presence/all_predictions.npy'.format(fp=fp), allow_pickle=True).item()
    # land_classifications = np.load('{fp}/lulc/all_lulc_predictions.npy'.format(fp=fp), allow_pickle=True).item()
    #
    # lulc_parth_values = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    #
    #
    # parth = parthenium[date]
    # lulc = land_classifications[date]
    #
    # for p_obs, l_obs in zip(parth.flatten(), lulc.flatten()):
    #     if l_obs > -1:
    #         lulc_parth_values[l_obs].append(p_obs)
    #
    # np.save('{fp}/parth_lulc_stats_{date}.npy'.format(fp=fp, date=date), lulc_parth_values)

    numbers = np.load('{fp}/mixed_data/parth_lulc_stats_{date}.npy'.format(fp=fp, date=date), allow_pickle=True).item()

    num_images = 0
    figure = plt.figure(figsize=(19.2, 10.8))

    for land_type in range(8):
        data = numbers[land_type]
        num_images += 1

        axes = figure.add_subplot(2, 4, num_images)
        axes.title.set_text("Class " + str(land_type))
        axes.hist(data, bins=50, histtype='bar', label='Class ' + str(land_type))
        axes.legend(loc='upper right')

    plt.suptitle('Parthenium prediction average per class 2019-06-20')
    figure.tight_layout()
    # plt.show()
    plt.savefig('{fp}/plots/parth_lulc_histogram_{date}.png'.format(fp=fp, date=date))


def binarise_data():
    all_binaries = np.load('{fp}/parth_presence/binary_data_0.5/all_binaries.npy'.format(fp=fp),
                           allow_pickle=True).item()

    binary_out = {
        "dateTime": [],
        "total_parth": [],
        "total_clear": [],
        "pc_parth": [],
        "pc_clear": [],
    }

    total = all_binaries['2018-12-27'].size

    for date in all_binaries.keys():
        binary_out["dateTime"].append(dates.datestr2num(date))

        data = all_binaries[date]
        parth = np.sum(data)
        clear = total - parth

        binary_out["total_parth"].append(parth)
        binary_out["total_clear"].append(clear)

        pc_parth = parth / float(total)
        pc_clear = clear / float(total)

        binary_out["pc_parth"].append(pc_parth)
        binary_out["pc_clear"].append(pc_clear)

    frame = pd.DataFrame(binary_out)
    print(frame)

    frame.to_pickle('{fp}/parth_presence/binary_data_0.5/binary_totals.pkl'.format(fp=fp))


def cloud_cover():
    scl = np.load('{fp}/raw_data/scl/2022_scl.npy'.format(fp=fp), allow_pickle=True).item()
    cloud_cover = np.load('{fp}/cloud_cover.npy'.format(fp=fp), allow_pickle=True).item()

    for date in scl.keys():
        mask = get_data.cloud_water_mask(scl[date])
        ratio = np.count_nonzero(mask == 0) / mask.size
        cloud_cover.update({date: ratio})

    np.save('{fp}/cloud_cover.npy'.format(fp=fp), cloud_cover)


def monthly_coverage():
    binaries = np.load('{fp}/parth_presence/binary_data_0.5/all_binaries.npy'.format(fp=fp), allow_pickle=True).item()
    clouds = np.load('{fp}/cloud_cover.npy'.format(fp=fp), allow_pickle=True).item()

    month_agg = {
        2018: {12: []},
        2019: {
            1: [], 2: [], 3: [], 4: [], 5: [], 6: [],
            7: [], 8: [], 9: [], 10: [], 11: [], 12: []
        },
        2020: {
            1: [], 2: [], 3: [], 4: [], 5: [], 6: [],
            7: [], 8: [], 9: [], 10: [], 11: [], 12: []
        },
        2021: {
            1: [], 2: [], 3: [], 4: [], 5: [], 6: [],
            7: [], 8: [], 9: [], 10: [], 11: [], 12: []
        },
        2022: {1: []}
    }

    for date in binaries.keys():
        if clouds[date] < 0.1:
            bin = binaries[date]
            pc_parth = np.sum(bin) / bin.size
            year = int(date.split('-')[0])
            month = int(date.split('-')[1])
            month_agg[year][month].append(pc_parth)

    parth_cover = {
        "year": [],
        "month": [],
        "pc_parth": []
    }

    for year in month_agg.keys():
        for month in month_agg[year].keys():
            percentages = month_agg[year][month]

            if len(percentages) > 0:
                parth_cover["year"].append(year)
                parth_cover["month"].append(month)
                parth_cover["pc_parth"].append(np.mean(percentages))

    frame = pd.DataFrame(parth_cover)
    frame.to_pickle('{fp}/monthly_parth_cover_0.5_10pc_cloud.pkl'.format(fp=fp))


def avg_monthly_coverage():
    data = pickle.load(open('{fp}/monthly_parth_cover_0.5_10pc_cloud.pkl'.format(fp=fp), 'rb'))
    averages = {}

    for month in range(1, 13):
        all_months = data[data['month'] == month]['pc_parth'].tolist()
        avg = np.mean(all_months)
        averages[month] = avg

    np.save('{fp}/avg_monthly_parth_cover_0.5_10pc_cloud.npy'.format(fp=fp), averages)


def import_csv():
    fp = '/scratch/nas_spiders/hillston/parthenium_simulation/data_overview'

    # prec = pd.read_csv('{fp}/precipitation.csv'.format(fp=fp))
    temp = pd.read_csv('{fp}/temperature.csv'.format(fp=fp))

    month_dict = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                  "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

    print(temp)

    date_list = temp["system:time_start"].tolist()

    days = []
    months = []
    years = []
    dateTimes = []

    for date in date_list:
        day = date.split(" ")[1].replace(',', '')
        month = month_dict[date.split(" ")[0]]
        year = date.split(' ')[2]
        dateTime = dates.datestr2num(str(month) + '-' + str(day) + '-' + + str(year))

        days.append(day)
        months.append(month)
        years.append(year)
        dateTimes.append(dateTime)

    means = np.asarray(temp["LST_Day_1km_mean"].tolist())
    mean_list = []
    for mean in means:
        mean_list.append((float(mean.replace(',', '')) * 0.02) - 273.15)

    stds = np.asarray(temp["LST_Day_1km_stdDev"].tolist())
    std_list = []
    for std in stds:
        std_list.append(std * 0.02)

    new_temp_dict = {
        "day": days,
        "month": months,
        "year": years,
        "dateTime": dateTimes,
        "temp_mean": mean_list,
        "temp_std": std_list,
    }

    frame = pd.DataFrame(new_temp_dict)

    print(frame)

    frame.to_pickle('{fp}/temperature.pkl'.format(fp=fp))

