from sklearn.cluster import KMeans
import numpy as np
from sklearn.impute import SimpleImputer

from imaging import get_data


def train_kmeans(training_set, features, clusters, nan_strategy):
    shape = np.shape(training_set)
    flattened_image = np.prod(shape[:-1]) if features > 1 else np.prod(shape)
    training_set = training_set.reshape(flattened_image, features)
    training_set = handle_nans(training_set, nan_strategy)
    classifier = KMeans(n_clusters=clusters).fit(training_set)
    return classifier


def run_classification(classifier, test_set, nan_strategy):
    shape = np.shape(test_set)
    test_set = test_set.reshape(np.prod(shape[:-1]), shape[-1])
    test_set = handle_nans(test_set, nan_strategy)
    return classifier.predict(test_set).reshape(shape[0], shape[1])


def handle_nans(data_set, nan_strategy):
    if nan_strategy == 'avg':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp = imp.fit(data_set)
        return imp.transform(data_set)
    elif nan_strategy == 'zero':
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
        imp = imp.fit(data_set)
        return imp.transform(data_set)
    elif nan_strategy == 'remove':
        return data_set[np.transpose(np.invert(np.isnan(data_set)))[0]]
    else:
        return data_set


def kmeans_single(raw_data, date, features):
    training_set = get_data.all_bands_by_date(raw_data, date)

    kmeans = train_kmeans(training_set)

    test_set = get_data.all_bands_by_date(raw_data, date)
    test_shape = np.shape(test_set)
    test_set = test_set.reshape(np.prod(test_shape[:-1]), features)

    return run_classification(kmeans, test_set)


def kmeans(raw_data, dates, features):
    training_dates = dates[:-1]

    training_set = []
    for date in training_dates:
        training_set.append(get_data.all_bands_by_date(raw_data, date))
    training_set = np.array(training_set)

    kmeans = train_kmeans(training_set)

    test_date = dates[-1:]
    test_set = get_data.all_bands_by_date(raw_data, test_date)
    test_shape = np.shape(test_set)
    test_set = test_set.reshape(np.prod(test_shape[:-2]), features)

    return run_classification(kmeans, test_set)


def knee_plot(raw_data, image_size, dates):
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt

    test_set = get_data.all_bands_by_date(raw_data, dates[0])
    test_set = test_set.reshape(np.prod(image_size), 10)

    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, 20)

    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(test_set)
        kmeanModel.fit(test_set)

        distortions.append(sum(np.min(cdist(test_set, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / test_set.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(test_set, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / test_set.shape[0]
        mapping2[k] = kmeanModel.inertia_

    np.save('./imaging/classifiers/knee_plot_distortions.npy', distortions)
    np.save('./imaging/classifiers/knee_plot_inertias.npy', inertias)

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.savefig('./imaging/plots/knee_plots/distortion.png')
    plt.close

    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.savefig('./imaging/plots/knee_plots/inertias.png')


class Classify:
    pass
