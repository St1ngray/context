from numpy import add
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
                          select_feature_automatically=False,
                          select_sample_automatically=False,
                          feature_normalization_method=None,
                          sample_normalization_method=None,
                          combining_function=add):

    if features is None:

        features = _select_elements_by_context_summary(
            feature_1d_context_matrix,
            select_context,
            n_top=n_top_feature,
            select_automatically=select_feature_automatically)

    if samples is None:

        samples = _select_elements_by_context_summary(
            sample_1d_context_matrix,
            select_context,
            n_top=n_top_sample,
            select_automatically=select_sample_automatically)

    feature_signal_matrix = _make_1d_signal_matrix(
        feature_1d_context_matrix.loc[features, samples], select_context)

    sample_signal_matrix = _make_1d_signal_matrix(
        sample_1d_context_matrix.loc[samples, features], select_context)

    if feature_normalization_method is not None:

        feature_signal_matrix = DataFrame(
            normalize_nd_array(
                feature_signal_matrix.values,
                feature_normalization_method,
                1,
                raise_for_bad_value=False), feature_signal_matrix.index,
            feature_signal_matrix.columns)

        features_without_signal = feature_signal_matrix.index[
            feature_signal_matrix.isna().all(axis=1)]

        if len(features_without_signal):

            print('Setting feature signals to 0 for: {} ...'.format(
                features_without_signal))

            feature_signal_matrix.loc[features_without_signal] = 0

    if sample_normalization_method is not None:

        sample_signal_matrix = DataFrame(
            normalize_nd_array(
                sample_signal_matrix.values,
                sample_normalization_method,
                1,
                raise_for_bad_value=False), sample_signal_matrix.index,
            sample_signal_matrix.columns)

        samples_without_signal = sample_signal_matrix.index[
            sample_signal_matrix.isna().all(axis=1)]

        if len(samples_without_signal):

            print('Setting sample signals to 0 for: {} ...'.format(
                samples_without_signal))

            sample_signal_matrix.loc[samples_without_signal] = 0

    return combining_function(feature_signal_matrix, sample_signal_matrix.T)
