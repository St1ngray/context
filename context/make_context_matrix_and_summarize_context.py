from os.path import join

from pandas import DataFrame, Series, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .compute_context import compute_context
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def make_context_matrix_and_summarize_context(
        matrix,
        n_job=1,
        skew_t_pdf_fit_parameter=None,
        fit_fixed_location=None,
        fit_fixed_scale=None,
        fit_initial_location=None,
        fit_initial_scale=None,
        n_grid=3000,
        degree_of_freedom_for_tail_reduction=10e8,
        global_location=None,
        global_scale=None,
        directory_path=None):
    """
    Make context matrix and summarize context.
    Arguments:
        matrix (DataFrame): (n_feature, n_sample, )
        n_job (int):
        skew_t_pdf_fit_parameter (DataFrame):
        fit_fixed_location (float):
        fit_fixed_scale (float):
        fit_initial_location (float):
        fit_initial_scale (float):
        n_grid (int):
        degree_of_freedom_for_tail_reduction (float):
        global_location (float):
        global_scale (float):
        directory_path (str):
    Returns:
        DataFrame: (n_feature, n_sample, )
        DataFrame: (n_feature, 2, )
    """

    returns = multiprocess(_make_context_matrix_and_summarize_context, ((
        matrix_,
        skew_t_pdf_fit_parameter,
        fit_fixed_location,
        fit_fixed_scale,
        fit_initial_location,
        fit_initial_scale,
        n_grid,
        degree_of_freedom_for_tail_reduction,
        global_location,
        global_scale, ) for matrix_ in split_df(matrix, n_job)), n_job)

    context_matrix = concat((r[0] for r in returns))
    context_summary = concat((r[1] for r in returns))

    if directory_path:
        establish_path(directory_path, 'directory')

        context_matrix.to_csv(
            join(directory_path, 'context_matrix.tsv'), sep='\t')

        context_summary.to_csv(
            join(directory_path, 'context_summary.tsv'), header=True, sep='\t')

    return context_matrix, context_summary


def _make_context_matrix_and_summarize_context(
        matrix, skew_t_pdf_fit_parameter, fit_fixed_location, fit_fixed_scale,
        fit_initial_location, fit_initial_scale, n_grid,
        degree_of_freedom_for_tail_reduction, global_location, global_scale):
    """
    Make context matrix and summarize context.
    Arguments:
        matrix (DataFrame): (n_feature, n_sample, )
        skew_t_pdf_fit_parameter (DataFrame):
        fit_fixed_location (float):
        fit_fixed_scale (float):
        fit_initial_location (float):
        fit_initial_scale (float):
        n_grid (int):
        degree_of_freedom_for_tail_reduction (float):
        global_location (float):
        global_scale (float):
    Returns:
        DataFrame: (n_feature, n_sample, )
        DataFrame: (n_feature, 2, )
    """

    skew_t_model = ACSkewT_gen()

    context_matrix = DataFrame(
        index=matrix.index, columns=matrix.columns, dtype=float)
    context_matrix.index.name = 'Feature'

    context_summary = DataFrame(
        index=context_matrix.index,
        columns=(
            'Negative Context Summary',
            'Positive Context Summary', ),
        dtype=float)

    n_per_log = max(matrix.shape[0] // 10, 1)

    for i, (
            index,
            vector, ) in enumerate(matrix.iterrows()):

        if i % n_per_log == 0:
            print('({}/{}) {} ...'.format(i + 1, matrix.shape[0], index))

        if skew_t_pdf_fit_parameter is None:
            location = scale = degree_of_freedom = shape = None
        else:
            location, scale, degree_of_freedom, shape = skew_t_pdf_fit_parameter.loc[
                index, [
                    'Location',
                    'Scale',
                    'Degree of Freedom',
                    'Shape',
                ]]

        context_dict = compute_context(
            vector,
            skew_t_model=skew_t_model,
            location=location,
            scale=scale,
            degree_of_freedom=degree_of_freedom,
            shape=shape,
            fit_fixed_location=fit_fixed_location,
            fit_fixed_scale=fit_fixed_scale,
            fit_initial_location=fit_initial_location,
            fit_initial_scale=fit_initial_scale,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=
            degree_of_freedom_for_tail_reduction,
            global_location=global_location,
            global_scale=global_scale)

        context_matrix.loc[index] = context_dict['context_indices_like_array']

        context_summary.loc[index] = context_dict[
            'negative_context_summary'], context_dict[
                'positive_context_summary']

    return context_matrix, context_summary
