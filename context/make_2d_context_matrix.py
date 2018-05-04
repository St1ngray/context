from os.path import join

from pandas import DataFrame

from .make_1d_context_matrix import make_1d_context_matrix
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array


def make_2d_context_matrix(matrix,
                           n_job=1,
                           skew_t_pdf_fit_parameter_0=None,
                           skew_t_pdf_fit_parameter_1=None,
                           n_grid=3000,
                           degree_of_freedom_for_tail_reduction=10e12,
                           global_location=None,
                           global_scale=None,
                           select_context=None,
                           combination_method='*',
                           directory_path=None):

    _1d_context_matrix_0 = _select_context_and_normalize_1d_context_matrix(
        make_1d_context_matrix(
            matrix,
            n_job=n_job,
            skew_t_pdf_fit_parameter=skew_t_pdf_fit_parameter_0,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=
            degree_of_freedom_for_tail_reduction,
            global_location=global_location,
            global_scale=global_scale), select_context)

    _1d_context_matrix_1 = _select_context_and_normalize_1d_context_matrix(
        make_1d_context_matrix(
            matrix.T,
            n_job=n_job,
            skew_t_pdf_fit_parameter=skew_t_pdf_fit_parameter_1,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=
            degree_of_freedom_for_tail_reduction,
            global_location=global_location,
            global_scale=global_scale), select_context).T

    if combination_method not in ('*', '+'):

        raise ValueError(
            'Unknown combination_method: {}.'.format(combination_method))

    if combination_method == '*':

        _2d_context_matrix = _1d_context_matrix_0 * _1d_context_matrix_1

    elif combination_method == '+':

        _2d_context_matrix = _1d_context_matrix_0 + _1d_context_matrix_1

    _2d_context_matrix.to_csv(
        join(directory_path, '2d_context_matrix.tsv'), sep='\t')

    return _2d_context_matrix


def _select_context_and_normalize_1d_context_matrix(_1d_context_matrix,
                                                    select_context):

    if select_context == '-_context':

        _1d_context_matrix[0 < _1d_context_matrix] = 0

        _1d_context_matrix *= -1

    elif select_context == '+_context':

        _1d_context_matrix[_1d_context_matrix < 0] = 0

    return DataFrame(
        normalize_2d_array(
            _1d_context_matrix.values, '0-1', 1, ignore_bad_value=True),
        index=_1d_context_matrix.index,
        columns=_1d_context_matrix.columns)
