from os.path import join

from pandas import Series

from .plot.plot.plot_points import plot_points
from .summarize_context_indices import summarize_context_indices
from .support.support.path import establish_path


def summarize_context_matrix_by_feature(context__feature_x_sample,
                                        fit_skew_t_pdf__feature_x_parameter,
                                        log=False,
                                        n_extreme_to_print=10,
                                        directory_path=None):
    """
    Summarize context matrix by feature.
    Arguments:
        context__feature_x_sample (DataFrame): (n_feature, n_sample)
        fit_skew_t_pdf__feature_x_parameter (DataFrame):
        log (bool): whether to log progress
        n_extreme_to_print (int): the number of extreme features to plot
        directory_path (str): where outputs are saved
    Returns:
        Series: (n_feature)
    """

    feature_context_summary = Series(
        index=context__feature_x_sample.index,
        name='Context Summary',
        dtype='float')

    for i, (feature_index, feature_context_vector
            ) in enumerate(context__feature_x_sample.iterrows()):
        if log:
            print('({}/{}) {} ...'.format(
                i + 1, context__feature_x_sample.shape[0], feature_index))

        feature_context_summary[feature_index] = summarize_context_indices(
            feature_context_vector,
            fit_skew_t_pdf__feature_x_parameter.loc[feature_index, 'Shape'])

    feature_context_summary.sort_values(inplace=True)

    if directory_path:
        establish_path(directory_path, path_type='directory')
        feature_context_summary.to_csv(
            join(directory_path, 'feature_context_summary.tsv'),
            header=True,
            sep='\t')

    plot_points(
        range(feature_context_summary.size),
        feature_context_summary,
        title='Ranked Context Summary',
        xlabel='Rank',
        ylabel='Context Summary')

    print('=' * 80)
    print('Extreme {} Context Summary'.format(n_extreme_to_print))
    print('v' * 80)
    for g, cs in feature_context_summary[:n_extreme_to_print].items():
        print('{}\t{}'.format(g, cs))
    print('*' * 80)
    for g, cs in feature_context_summary[-n_extreme_to_print:].items():
        print('{}\t{}'.format(g, cs))
    print('=' * 80)

    return feature_context_summary
