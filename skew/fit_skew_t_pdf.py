from os.path import join

from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_1d_array_to_skew_t_pdf import fit_1d_array_to_skew_t_pdf
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def fit_skew_t_pdf(feature_x_sample, n_job=1, directory_path=None):
    """
    Fit DataFrame rows (features) to skew-t PDF.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        n_job (int): number of jobs for parallel computing
        directory_path (str): where outputs are saved
    Returns:
        DataFrame: (n_feature, 5 [N, DF, Shape, Location, Scale])
    """

    skew_t_model = ACSkewT_gen()

    fit_skew_t_pdf__feature_x_parameter = concat(
        multiprocess(_fit_skew_t_pdf, [[df, skew_t_model]
                                       for df in split_df(
                                           feature_x_sample, n_job)], n_job))

    fit_skew_t_pdf__feature_x_parameter.sort_values('Shape', inplace=True)

    if directory_path:
        establish_path(directory_path, path_type='directory')
        fit_skew_t_pdf__feature_x_parameter.to_csv(
            join(directory_path, 'fit_skew_t_pdf__feature_x_parameter.tsv'),
            sep='\t')

    return fit_skew_t_pdf__feature_x_parameter


def _fit_skew_t_pdf(feature_x_sample, skew_t_model):
    """
    Fit DataFrame rows (features) to skew-t PDF.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        skew_t_model (statsmodels.sandbox.distributions.extras.ACSkewT_gen):
    Returns:
        DataFrame: (n_feature, 5 [N, DF, Shape, Location, Scale])
    """

    fit_skew_t_pdf__feature_x_parameter = DataFrame(
        index=feature_x_sample.index,
        columns=['N', 'Location', 'Scale', 'DF', 'Shape'])
    fit_skew_t_pdf__feature_x_parameter.index.name = 'Feature'

    for i, (feature_index,
            feature_vector) in enumerate(feature_x_sample.iterrows()):
        print('({}/{}) {} ...'.format(i + 1, feature_x_sample.shape[0],
                                      feature_index))

        fit_skew_t_pdf__feature_x_parameter.loc[
            feature_index] = feature_vector.size, *fit_1d_array_to_skew_t_pdf(
                feature_vector, skew_t_model=skew_t_model)

    return fit_skew_t_pdf__feature_x_parameter
