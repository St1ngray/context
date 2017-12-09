from os.path import join

from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def fit_skew_t_pdf(feature_x_sample, n_job=1, directory_path=None):
    """
    Fit skew-t PDF to the distribution of each feature.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        n_job (int): number of jobs for parallel computing
        directory_path (str): where outputs are saved
    Returns:
        DataFrame: (n_feature, 5 [N, DF, Shape, Location, Scale])
    """

    print(
        'Fitting skew-t PDF to the distribution of each feature across {} jobs ...'.
        format(n_job))

    feature_x_feature = concat(
        multiprocess(_fit_skew_t_pdf, [[df]
                                       for df in split_df(
                                           feature_x_sample, n_job)], n_job))

    feature_x_feature.sort_values('Shape', inplace=True)

    if directory_path:
        establish_path(directory_path, path_type='directory')
        feature_x_feature.to_csv(
            join(directory_path, 'fit_skew_t_pdf.tsv'), sep='\t')

    return feature_x_feature


def _fit_skew_t_pdf(feature_x_sample):
    """
    Fit skew-t PDF to the distribution of each feature.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
    Returns:
        DataFrame: (n_feature, 5 [N, DF, Shape, Location, Scale])
    """

    feature_x_feature = DataFrame(
        index=feature_x_sample.index,
        columns=['N', 'Location', 'Scale', 'DF', 'Shape'])
    feature_x_feature.index.name = 'Feature'

    for i, (f_i, f_v) in enumerate(feature_x_sample.iterrows()):
        print('({}/{}) {} ...'.format(i + 1, feature_x_sample.shape[0], f_i))

        skew_t = ACSkewT_gen()
        f_v.dropna(inplace=True)
        df, shape, location, scale = skew_t.fit(f_v)
        feature_x_feature.loc[f_i] = f_v.size, location, scale, df, shape

    return feature_x_feature
