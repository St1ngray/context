from pandas import DataFrame

from .make_1d_signal_matrix import make_1d_signal_matrix
from .nd_array.nd_array.normalize_nd_array import normalize_nd_array


def make_2d_signal_matrix(feature_1d_context_matrix, sample_1d_context_matrix,
                          select_context):

    feature_signal_matrix = make_1d_signal_matrix(feature_1d_context_matrix,
                                                  select_context)

    feature_signal_matrix = DataFrame(
        normalize_nd_array(
            feature_signal_matrix.values, '0-1', 1, raise_for_bad_value=False),
        feature_signal_matrix.index, feature_signal_matrix.columns)

    sample_signal_matrix = make_1d_signal_matrix(sample_1d_context_matrix,
                                                 select_context)

    sample_signal_matrix = DataFrame(
        normalize_nd_array(
            sample_signal_matrix.values, '0-1', 1, raise_for_bad_value=False),
        sample_signal_matrix.index, sample_signal_matrix.columns)

    return feature_signal_matrix * sample_signal_matrix.T
