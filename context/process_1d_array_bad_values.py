from numpy import nanmean

from .nd_array.nd_array.check_nd_array_for_bad_value import \
    check_nd_array_for_bad_value


def process_1d_array_bad_values(_1d_array):

    _1d_array = _1d_array.copy()

    is_bad_value = check_nd_array_for_bad_value(_1d_array, raise_=False)

    if is_bad_value.all():

        raise ValueError('_1d_array has only bad value.')

    elif is_bad_value.any():

        _1d_array[is_bad_value] = nanmean(_1d_array)

    return _1d_array
