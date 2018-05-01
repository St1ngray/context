from warnings import warn

from numpy import isnan, nanmean


def process_1d_array_bad_values(_1d_array):

    _1d_array = _1d_array.copy()

    is_nan = isnan(_1d_array)

    if is_nan.all():

        raise ValueError('_1d_array has only nan.')

    elif is_nan.any():

        warn('Replacing nan with mean ...')

        _1d_array[is_nan] = nanmean(_1d_array)

    return _1d_array
