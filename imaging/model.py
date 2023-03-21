from sklearn.cluster import KMeans
from imaging import indices


def kmeans_one_feature(raw_data, date_range):
    valid_indices = [1, 2, 4, 5, 6, 8]
    training_dates = [date_range[i] for i in valid_indices]
    test_date = date_range[9]

    training_set = indices.ndvi(raw_data, training_dates[0])
    test_set = indices.ndvi(raw_data, test_date)

    training_set = training_set.flatten().reshape(-1, 1)
    test_set = test_set.flatten().reshape(-1, 1)

    return KMeans(n_clusters=2).fit(training_set)


class Model:
    pass