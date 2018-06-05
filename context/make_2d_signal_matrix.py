from numpy import multiply
from pandas import DataFrame

from ._make_1d_signal_matrix import _make_1d_signal_matrix
from ._select_elements_by_context_summary import \
    _select_elements_by_context_summary
from .nd_array.nd_array.normalize_nd_array import normalize_nd_array


def make_2d_signal_matrix(feature_1d_context_matrix,
                          sample_1d_context_matrix,
                          select_context,
                          features=None,
                          samples=None,
                          n_top_feature=None,
                          n_top_sample=None,
                          select_automatically=False,
                          normalization_method='0-1',
                          combining_function=multiply):

    if features is None:

        features = _select_elements_by_context_summary(
            feature_1d_context_matrix,
            select_context,
            n_top=n_top_feature,
            select_automatically=select_automatically)

    if samples is None:

        samples = _select_elements_by_context_summary(
            sample_1d_context_matrix,
            select_context,
            n_top=n_top_sample,
            select_automatically=select_automatically)

    feature_signal_matrix = _make_1d_signal_matrix(
        feature_1d_context_matrix.loc[features, samples], select_context)

    sample_signal_matrix = _make_1d_signal_matrix(
        sample_1d_context_matrix.loc[samples, features], select_context)

    if normalization_method is not None:

        try:

            feature_signal_matrix = DataFrame(
                normalize_nd_array(
                    feature_signal_matrix.values,
                    normalization_method,
                    1,
                    raise_for_bad_value=False), feature_signal_matrix.index,
                feature_signal_matrix.columns)

        except ValueError as exception:

            raise ValueError(
                'Normalizing feature_signal_matrix failed.\nException: {}.\nTry setting normalization_method=None.'.
                format(exception))

        try:

            sample_signal_matrix = DataFrame(
                normalize_nd_array(
                    sample_signal_matrix.values,
                    normalization_method,
                    1,
                    raise_for_bad_value=False), sample_signal_matrix.index,
                sample_signal_matrix.columns)

        except ValueError as exception:

            raise ValueError(
                'Normalizing sample_signal_matrix failed.\nException: {}.\nTry setting normalization_method=None.'.
                format(exception))

    return combining_function(feature_signal_matrix, sample_signal_matrix.T)
