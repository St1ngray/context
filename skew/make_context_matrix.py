from os.path import join

from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .compute_context_indices import compute_context_indices
from .fit_1d_array_to_skew_t_pdf import fit_1d_array_to_skew_t_pdf
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def make_context_matrix(feature_x_sample,
                        fit_skew_t_pdf__feature_x_parameter=None,
                        n_grid=3000,
                        n_job=1,
                        directory_path=None):
    """
    Make context matrix.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        fit_skew_t_pdf__feature_x_parameter (DataFrame):
        n_grid (int):
        n_job (int): number of jobs for parallel computing
        directory_path (str): where outputs are saved
    Returns:
        DataFrame: (n_feature, n_sample)
    """

    skew_t_model = ACSkewT_gen()

    context__feature_x_sample = concat(
        multiprocess(_make_context_matrix, [[
            df, skew_t_model, fit_skew_t_pdf__feature_x_parameter, n_grid
        ] for df in split_df(feature_x_sample, n_job)], n_job))

    if directory_path:
        establish_path(directory_path, path_type='directory')
        context__feature_x_sample.to_csv(
            join(directory_path, 'context__feature_x_sample.tsv'), sep='\t')

    return context__feature_x_sample


def _make_context_matrix(feature_x_sample, skew_t_model,
                         fit_skew_t_pdf__feature_x_parameter, n_grid):
    """
    Make context matrix.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        skew_t_model (statsmodels.sandbox.distributions.extras.ACSkewT_gen):
        fit_skew_t_pdf__feature_x_parameter (DataFrame):
        n_grid (int):
    Returns:
        DataFrame: (n_feature, n_sample)
    """

    context__feature_x_sample = DataFrame(
        index=feature_x_sample.index, columns=feature_x_sample.columns)

    for i, (feature_index,
            feature_vector) in enumerate(feature_x_sample.iterrows()):

        if fit_skew_t_pdf__feature_x_parameter is None:
            location, scale, df, shape = fit_1d_array_to_skew_t_pdf(
                feature_vector, skew_t_model=skew_t_model)
        else:
            location, scale, df, shape = fit_skew_t_pdf__feature_x_parameter.loc[
                i, ['Location', 'Scale', 'DF', 'Shape']]

        context__feature_x_sample.loc[feature_index] = compute_context_indices(
            feature_vector,
            skew_t_model=skew_t_model,
            n_grid=n_grid,
            location=location,
            scale=scale,
            df=df,
            shape=shape)

    return context__feature_x_sample