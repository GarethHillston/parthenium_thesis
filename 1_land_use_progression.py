import pickle
import xarray as xr
import numpy as np
import utilities as util
from imaging import get_data, classify, render

filePath = '/scratch/nas_bridle/sentinel/shared/rawalpindi_1.nc'
raw_data = xr.open_dataset(filePath)
date_times = util.get_dates(raw_data)
dates = np.array([d.split('T')[0] for d in date_times])

output_folder = './imaging/plots/land_classifications/19_20_rawalpindi'
cmap = render.eight_colours

training_data = []
test_data = {}

# get all images over time series
for date_time in date_times:
    all_bands_for_date = get_data.all_bands_by_date(raw_data, date_time)
    # cover = get_data.cloud_water_mask(raw_data, date_time)
    # nans = cover == 0
    # all_bands_for_date[nans] = np.nan
    training_data.append(all_bands_for_date)
training_data = np.array(training_data)
print(np.shape(training_data))
# # np.save('lulc_parth_combined_model/cloud_covered_land.npy', training_data)
# training_data = np.load('lulc_parth_combined_model/cloud_covered_land.npy')
#
# # create classifier on them
# for date_time in date_times:
#     all_bands_for_date = get_data.all_bands_by_date(raw_data, date_time)
#     test_data[date_time.split('T')[0]] = all_bands_for_date
#
# features = 10
# kmeans = classify.train_kmeans_nans(training_data, features, 8)
# pickle.dump(kmeans, open('lulc_parth_combined_model/LULC_test.pkl', 'wb'))

# kmeans = pickle.load(open('imaging/classifiers/LULC_test.pkl', 'rb'))
# features = 10
# # run classifier
# results = []
# image_shape = np.shape(test_data[dates[0]])
#
# for date in dates:
#     classification = classify.run_classification(kmeans, test_data[date])
#     results.append(classification)
#     render.single_plot(classification, date, cmap, '{folder}'.format(folder=output_folder))
#
# results = np.array(results)
#
# # save output
# np.save('{folder}/19_20_rawalpindi_land_classes.npy'.format(folder=output_folder), results)
