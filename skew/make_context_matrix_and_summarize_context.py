from os.path import join

from pandas import DataFrame, Series, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .compute_context import compute_context
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def make_context_matrix_and_summarize_context(
        feature_x_sample,
        fit_skew_t_pdf__feature_x_parameter=None,
        n_grid=3000,
        compute_context_method='tail_reduction_reflection',
        degrees_of_freedom_for_tail_reduction=10e8,
        summarize_context_by='absolute_value_weighted_context',
        summarize_context_side='shape_side',
        n_job=1,
        log=False,
        directory_path=None):
    """
    Make context matrix and summarize context.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        fit_skew_t_pdf__feature_x_parameter (DataFrame):
        n_grid (int):
        compute_context_method (str): 'tail_reduction_reflection' |
            'tail_reduction' | 'reflection' |
        degrees_of_freedom_for_tail_reduction (number):
        summarize_context_by (str): 'absolute_value_weighted_context' |
            'context'
        summarize_context_side (str): 'shape_side' | 'both_sides'
        n_job (int): number of jobs for parallel computing
        log (bool): whether to log progress
        directory_path (str): where outputs are saved
    Returns:
        DataFrame: (n_feature, n_sample)
        Series: (n_feature)
    """

    returns = multiprocess(_make_context_matrix_and_summarize_context, [[
        df, fit_skew_t_pdf__feature_x_parameter, n_grid,
        compute_context_method, degrees_of_freedom_for_tail_reduction,
        summarize_context_by, summarize_context_side, log
    ] for df in split_df(feature_x_sample, n_job)], n_job)

    context__feature_x_sample = concat([r[0] for r in returns])
    feature_context_summary = concat([r[1] for r in returns]).sort_values()

    if directory_path:
        establish_path(directory_path, path_type='directory')

        context__feature_x_sample.to_csv(
            join(directory_path, 'context__feature_x_sample.tsv'), sep='\t')

        feature_context_summary.to_csv(
            join(directory_path, 'feature_context_summary.tsv'),
            header=True,
            sep='\t')

    return context__feature_x_sample, feature_context_summary


def _make_context_matrix_and_summarize_context(
        feature_x_sample, fit_skew_t_pdf__feature_x_parameter, n_grid,
        compute_context_method, degrees_of_freedom_for_tail_reduction,
        summarize_context_by, summarize_context_side, log):
    """
    Make context matrix and summarize context.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        fit_skew_t_pdf__feature_x_parameter (DataFrame):
        n_grid (int):
        compute_context_method (str): 'tail_reduction' | 'reflection' |
            'tail_reduction_reflection'
        degrees_of_freedom_for_tail_reduction (number):
        summarize_context_by (str): 'context' |
            'absolute_value_weighted_context'
        summarize_context_side (str): 'shape_side' | 'both_sides'
        log (bool): whether to log progress
    Returns:
        DataFrame: (n_feature, n_sample)
        Series: (n_feature)
    """

    skew_t_model = ACSkewT_gen()

    context__feature_x_sample = DataFrame(
        index=feature_x_sample.index,
        columns=feature_x_sample.columns,
        dtype='float')
    context__feature_x_sample.index.name = 'Feature'

    feature_context_summary = Series(
        index=context__feature_x_sample.index,
        name='Context Summary',
        dtype='float')

    for i, (feature_index,
            feature_vector) in enumerate(feature_x_sample.iterrows()):
        if log:
            print('({}/{}) {} ...'.format(i + 1, feature_x_sample.shape[0],
                                          feature_index))

        if fit_skew_t_pdf__feature_x_parameter is None:
            location = scale = df = shape = None
        else:
            location, scale, df, shape = fit_skew_t_pdf__feature_x_parameter.loc[
                feature_index, ['Location', 'Scale', 'DF', 'Shape']]

        context_dict = compute_context(
            feature_vector,
            skew_t_model=skew_t_model,
            n_grid=n_grid,
            location=location,
            scale=scale,
            df=df,
            shape=shape,
            compute_context_method=compute_context_method,
            degrees_of_freedom_for_tail_reduction=
            degrees_of_freedom_for_tail_reduction,
            summarize_context_by=summarize_context_by,
            summarize_context_side=summarize_context_side)

        context__feature_x_sample.loc[feature_index] = context_dict[
            'context_indices_like_array']

        feature_context_summary[feature_index] = context_dict[
            'context_summary']

    return context__feature_x_sample, feature_context_summary
