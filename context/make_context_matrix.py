from pandas import DataFrame

from .make_context_matrix_and_summarize_context_by_row import \
    make_context_matrix_and_summarize_context_by_row
from .nd_array.nd_array.normalize_2d_array import normalize_2d_array


def make_context_matrix(matrix,
                        n_job=1,
                        skew_t_pdf_fit_parameter_by_row=None,
                        skew_t_pdf_fit_parameter_by_column=None,
                        n_grid=3000,
                        degree_of_freedom_for_tail_reduction=10e12,
                        global_location=None,
                        global_scale=None,
                        select_context=None,
                        combination_method='*',
                        directory_path=None):

    context_matrix_by_row = select_context_and_normalize_context_matrix_by_row(
        make_context_matrix_and_summarize_context_by_row(
            matrix,
            n_job=n_job,
            skew_t_pdf_fit_parameter=skew_t_pdf_fit_parameter_by_row,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=
            degree_of_freedom_for_tail_reduction,
            global_location=global_location,
            global_scale=global_scale)[0], select_context)

    context_matrix_by_column = select_context_and_normalize_context_matrix_by_row(
        make_context_matrix_and_summarize_context_by_row(
            matrix.T,
            n_job=n_job,
            skew_t_pdf_fit_parameter=skew_t_pdf_fit_parameter_by_column,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=
            degree_of_freedom_for_tail_reduction,
            global_location=global_location,
            global_scale=global_scale)[0], select_context).T

    if combination_method == '*':

        context_matrix = context_matrix_by_row * context_matrix_by_column

    elif combination_method == '+':

        context_matrix = context_matrix_by_row + context_matrix_by_column

    else:

        raise ValueError(
            'Unknown combination_method: {}.'.format(combination_method))

    return context_matrix


def select_context_and_normalize_context_matrix_by_row(context_matrix,
                                                       select_context):

    if select_context == '-_context':

        context_matrix[0 < context_matrix] = 0
        context_matrix *= -1

    elif select_context == '+_context':

        context_matrix[context_matrix < 0] = 0

    return DataFrame(
        normalize_2d_array(context_matrix.values, '0-1', 1),
        index=context_matrix.index,
        columns=context_matrix.columns)
