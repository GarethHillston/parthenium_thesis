import numpy as np
from imaging import render

predictions = np.load('modelling/progressions/2019_2020_predictions/all_predictions.npy')
predictions = np.transpose(predictions, (2, 0, 1))
predictions = predictions[:10]
# render.single_plot(parth_presence[10], 'Predictions', 'Inferno', '')

classifications = np.load('lulc_parth_combined_model/19_20_rawalpindi_land_classes.npy')

class_stats = [[], [], [], [], [], [], [], []]

for predict_date, classification_date in zip(predictions, classifications):
    for predict_row, class_row in zip(predict_date, classification_date):
        for parth, cluster in zip(predict_row, class_row):
            class_stats[cluster].append(parth)

class_stat_size = len(class_stats)
all_stats = {'mean': [], 'median': [], 'std': [], 'proportion': []}

for class_row, number in zip(class_stats, range(class_stat_size)):
    print('Class ', number)

    mean = np.nanmean(class_row)
    print('Mean   - ', mean)
    all_stats['mean'].append(mean)

    median = np.nanmedian(class_row)
    print('Median - ', median)
    all_stats['median'].append(median)

    std = np.nanstd(class_row)
    print('STD    - ', std)
    all_stats['std'].append(std)

    print('Total  - ', len(class_row))
    print()
total = 0
for class_row in class_stats:
    total += np.nansum(class_row)

proportions = np.empty(class_stat_size)

for class_row, i in zip(class_stats, range(class_stat_size)):
    proportions[i] = np.nansum(class_row) / total
    print('Class {num} - {proportion}'.format(num=i, proportion=proportions[i]))
print(np.sum(proportions))

all_stats['proportion'] = proportions
print(all_stats)

np.save('lulc_parth_combined_model/parth_lulc_proportions_19_20.npy', proportions)

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