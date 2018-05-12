from os.path import join

from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .compute_context import compute_context
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def make_1d_context_matrix(matrix,
                           n_job=1,
                           skew_t_pdf_fit_parameter=None,
                           n_grid=3000,
                           degree_of_freedom_for_tail_reduction=10e12,
                           global_location=None,
                           global_scale=None,
                           global_degree_of_freedom=None,
                           global_shape=None,
                           directory_path=None):

    _1d_context_matrix = concat(
        multiprocess(_make_1d_context_matrix,
                     ((matrix_, skew_t_pdf_fit_parameter, n_grid,
                       degree_of_freedom_for_tail_reduction, global_location,
                       global_scale, global_degree_of_freedom, global_shape)
                      for matrix_ in split_df(matrix, 0, n_job)), n_job))

    if directory_path is not None:

        establish_path(directory_path, 'directory')

        _1d_context_matrix.to_csv(
            join(directory_path, '1d_context_matrix.tsv'), sep='\t')

    return _1d_context_matrix


def _make_1d_context_matrix(matrix, skew_t_pdf_fit_parameter, n_grid,
                            degree_of_freedom_for_tail_reduction,
                            global_location, global_scale,
                            global_degree_of_freedom, global_shape):

    skew_t_model = ACSkewT_gen()

    _1d_context_matrix = DataFrame(
        index=matrix.index, columns=matrix.columns, dtype=float)

    n = matrix.shape[0]

    n_per_print = max(n // 10, 1)

    for i, (index, _1d_array) in enumerate(matrix.iterrows()):

        if i % n_per_print == 0:

            print('({}/{}) {} ...'.format(i + 1, n, index))

        if skew_t_pdf_fit_parameter is None:

            location = scale = degree_of_freedom = shape = None

        else:

            location, scale, degree_of_freedom, shape = skew_t_pdf_fit_parameter.loc[
                index, ['Location', 'Scale', 'Degree of Freedom', 'Shape']]

        _1d_context_matrix.loc[index] = compute_context(
            _1d_array,
            skew_t_model=skew_t_model,
            location=location,
            scale=scale,
            degree_of_freedom=degree_of_freedom,
            shape=shape,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=
            degree_of_freedom_for_tail_reduction,
            global_location=global_location,
            global_scale=global_scale,
            global_degree_of_freedom=global_degree_of_freedom,
            global_shape=global_shape)['context_indices_like_array']

    return _1d_context_matrix
