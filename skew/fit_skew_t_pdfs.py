from os.path import join

from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_skew_t_pdf import fit_skew_t_pdf
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def fit_skew_t_pdfs(feature_x_sample, n_job=1, log=False, directory_path=None):
    """
    Fit skew-t PDFs.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        n_job (int):
        log (bool):
        directory_path (str):
    Returns:
        DataFrame: (n_feature, 5 [N, Location, Scale, DF, Shape])
    """

    fit_skew_t_pdf__feature_x_parameter = concat(
        multiprocess(_fit_skew_t_pdfs, [[df, log]
                                        for df in split_df(
                                            feature_x_sample, n_job)], n_job))

    if directory_path:
        establish_path(directory_path, path_type='directory')

        fit_skew_t_pdf__feature_x_parameter.to_csv(
            join(directory_path, 'fit_skew_t_pdf__feature_x_parameter.tsv'),
            sep='\t')

    return fit_skew_t_pdf__feature_x_parameter


def _fit_skew_t_pdfs(feature_x_sample, log):
    """
    Fit skew-t PDFs.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        log (bool): whether to log progress
    Returns:
        DataFrame: (n_feature, 5 [N, Location, Scale, DF, Shape])
    """

    skew_t_model = ACSkewT_gen()

    fit_skew_t_pdf__feature_x_parameter = DataFrame(
        index=feature_x_sample.index,
        columns=['N', 'Location', 'Scale', 'DF', 'Shape'],
        dtype='float')
    fit_skew_t_pdf__feature_x_parameter.index.name = 'Feature'

    for i, (feature_index,
            feature_vector) in enumerate(feature_x_sample.iterrows()):
        if log:
            print('({}/{}) {} ...'.format(i + 1, feature_x_sample.shape[0],
                                          feature_index))

        fit_skew_t_pdf__feature_x_parameter.loc[
            feature_index] = fit_skew_t_pdf(
                feature_vector, skew_t_model=skew_t_model)

    return fit_skew_t_pdf__feature_x_parameter
