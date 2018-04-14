from warnings import warn

from numpy import isnan, nanmean


def process_array_1d_bad_values(array_1d):

    array_1d = array_1d.copy()

    is_nan = isnan(array_1d)

    if is_nan.all():
        raise ValueError('array_1d has only nan.')

    elif is_nan.any():
        warn('Replacing nan with mean ...')
        array_1d[is_nan] = nanmean(array_1d)

    return array_1d
