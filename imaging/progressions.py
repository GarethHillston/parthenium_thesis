import numpy as np


def binarise_progressions(dates, predictions_folder, binary_folder):
    for date in dates:
        prediction = np.load('{folder}/prediction_{date}.npy'.format(date=date, folder=predictions_folder))
        binarised_predict = np.zeros(np.shape(prediction), dtype=object)
        ones = prediction > 0.5
        binarised_predict[ones] = 1
        # nans = prediction == np.nan
        # binarised_predict[nans] = np.nan
        np.save('{folder}/prediction_{date}.npy'.format(date=date, folder=binary_folder), binarised_predict)


def unnan_binary_progression(progression):
    return_progression = np.zeros(np.shape(progression), dtype=int)
    ones = progression == 1
    return_progression[ones] = 1
    return return_progression


class Progressions:
    pass