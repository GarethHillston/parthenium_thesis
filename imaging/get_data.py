import numpy as np
from astropy.stats import sigma_clip


def normalise(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def nan_normalise(array):
    return (array - np.nanmin(array)) / (np.nanmax(array) - np.nanmin(array))


def by_band_and_date(raw_data, band, date):
    return raw_data.get(date).get(band).astype(np.float64)


def all_bands_by_date(raw_data, date):
    return raw_data.get(date).astype(np.float64)


def by_band_and_date_sigma_clip(raw_data, band, date):
    reflectance_data = raw_data.get(date).get(band).astype(np.float64)
    clipped_data = sigma_clip(reflectance_data, sigma_upper=3)
    return clipped_data


def by_band_and_date_manual_clip(raw_data, band, date):
    reflectance_data = raw_data.get(date).get(band).astype(np.float64)
    clipped_data = reflectance_data.clip(max=3500)
    return clipped_data


def cloud_water_mask(scl_data):
    # 4 = vegetation, 5 = bare soil, 7 = low probability clouds / unclassified
    condition = (scl_data == 4) | (scl_data == 5) | (scl_data == 7)
    return np.where(condition, 1, 0)


def cloud_mask(scl_data):
    # 4 = vegetation, 5 = bare soil, 6 = water, 7 = low probability clouds / unclassified, 11 = snow/ice
    condition = (scl_data == 4) | (scl_data == 5) | (scl_data == 6) | (scl_data == 7) | (scl_data == 11)
    return np.where(condition, 1, 0)


def light_mask(scl_data):
    # 1 = Saturated / Defective, 9 = High probability clouds, 10 = Cirrus clouds
    condition = (scl_data == 1) | (scl_data == 9) | (scl_data == 10)
    return np.where(condition, 0, 1)


def apply_standard_mask(ref, scl, mask_value):
    cover = cloud_water_mask(scl)
    mask = cover == 0
    ref[mask] = mask_value
    return ref


def apply_standard_mask_dict(ref, scl):
    mask_ref = {}

    for date in ref.keys():
        mask_ref[date] = apply_standard_mask(ref[date], scl[date], np.nan)

    return mask_ref


def get_year_from_dict(dict, year):
    out_dict = {}
    for date in dict.keys():
        if date.split('-')[0] == year:
            out_dict[date] = dict[date]
    return out_dict


def get_month_from_dict(dict, month):
    out_dict = {}
    for date in dict.keys():
        if date.split('-')[1] == month:
            out_dict[date] = dict[date]
    return out_dict


def year(date):
    return int(date.split('-')[0])


def month(date):
    return int(date.split('-')[1])


def day(date):
    return int(date.split('-')[2])


def generate_RGB(ref):
    red = ref[3]
    green = ref[2]
    blue = ref[1]

    norm_red = nan_normalise(red)
    norm_green = nan_normalise(green)
    norm_blue = nan_normalise(blue)

    return np.dstack((norm_red, norm_green, norm_blue))

    # return np.dstack((red, green, blue))


class GetData:
    pass
