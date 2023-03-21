import numpy as np
from imaging import get_data

bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']


def index(add_band, sub_band):
    return (add_band.astype(float) - sub_band.astype(float)) / (add_band + sub_band)


# https://custom-scripts.sentinel-hub.com/sentinel-2/ndvi/
def ndvi(ref_data):
    red = ref_data[2]
    nir = ref_data[6]

    return index(nir, red)


# https://custom-scripts.sentinel-hub.com/sentinel-2/gndvi/
def gndvi(raw_data, date):
    green = get_data.by_band_and_date(raw_data, 'B03', date)
    nir = get_data.by_band_and_date(raw_data, 'B08', date)

    return index(nir, green)


# https://custom-scripts.sentinel-hub.com/sentinel-2/ndmi/
def ndmi(raw_data, date):
    swir = get_data.by_band_and_date(raw_data, 'B11', date)
    nir = get_data.by_band_and_date(raw_data, 'B08', date)

    return index(nir, swir)


def bare_soil_index(raw_data, date):
    low_swir = get_data.by_band_and_date(raw_data, 'B11', date)
    red = get_data.by_band_and_date(raw_data, 'B04', date)
    nir = get_data.by_band_and_date(raw_data, 'B08', date)
    blue = get_data.by_band_and_date(raw_data, 'B02', date)

    soil_index = 2.5 * ((low_swir + red) - (nir + blue)) / ((low_swir + red) + (nir + blue))

    return soil_index


def __urban_pixel_value(i, j, ndvi, ndmi, soil, swir):
    if ndmi[i][j] > 0.2:
        return [0, 0.5, 1]
    elif swir[i][j] > 0.8 or ndvi[i][j] < 0.1:
        return [1, 1, 1]
    elif ndvi[i][j] > 0.2:
        return [0, 0.3 * ndvi[i][j], 0]
    else:
        return [soil[i][j], 0.2, 0]


def urban_classified(raw_data, date):
    ndvi_scores = ndvi(raw_data, date)
    ndmi_scores = ndmi(raw_data, date)
    soil_index = bare_soil_index(raw_data, date)
    low_swir = get_data.by_band_and_date(raw_data, 'B11', date)

    shape = np.shape(low_swir)
    classified_image = np.zeros((shape[0], shape[1], 3))

    for i in range(shape[0]):
        for j in range(shape[1]):
            classified_image[i][j] = __urban_pixel_value(i, j, ndvi_scores, ndmi_scores, soil_index, low_swir)

    return classified_image


class Indices:
    pass
