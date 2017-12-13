from os.path import join

from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .compute_context_indices import compute_context_indices
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def make_context_matrix(feature_x_sample,
                        fit_skew_t_pdf__feature_x_parameter=None,
                        n_grid=3000,
                        compute_context_indices_method='tail_reduction',
                        log=False,
                        n_job=1,
                        directory_path=None):
    """
    Make context matrix.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        fit_skew_t_pdf__feature_x_parameter (DataFrame):
        n_grid (int):
        compute_context_indices_method (str): 'tail_reduction' | 'reflection'
        log (bool): whether to log progress
        n_job (int): number of jobs for parallel computing
        directory_path (str): where outputs are saved
    Returns:
        DataFrame: (n_feature, n_sample)
    """

    context__feature_x_sample = concat(
        multiprocess(_make_context_matrix, [[
            df, fit_skew_t_pdf__feature_x_parameter, n_grid,
            compute_context_indices_method, log
        ] for df in split_df(feature_x_sample, n_job)], n_job))

    if directory_path:
        establish_path(directory_path, path_type='directory')
        context__feature_x_sample.to_csv(
            join(directory_path, 'context__feature_x_sample.tsv'), sep='\t')

    return context__feature_x_sample


def _make_context_matrix(feature_x_sample, fit_skew_t_pdf__feature_x_parameter,
                         n_grid, compute_context_indices_method, log):
    """
    Make context matrix.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        fit_skew_t_pdf__feature_x_parameter (DataFrame):
        n_grid (int):
        compute_context_indices_method (str): 'tail_reduction' | 'reflection'
        log (bool): whether to log progress
    Returns:
        DataFrame: (n_feature, n_sample)
    """

    skew_t_model = ACSkewT_gen()

    context__feature_x_sample = DataFrame(
        index=feature_x_sample.index,
        columns=feature_x_sample.columns,
        dtype='float')
    context__feature_x_sample.index.name = 'Feature'

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

        context__feature_x_sample.loc[feature_index] = compute_context_indices(
            feature_vector,
            skew_t_model=skew_t_model,
            n_grid=n_grid,
            location=location,
            scale=scale,
            df=df,
            shape=shape,
            compute_context_indices_method=compute_context_indices_method)[
                'context_indices_like_array']

    return context__feature_x_sample
