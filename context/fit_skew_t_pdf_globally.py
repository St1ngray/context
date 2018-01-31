from pandas import isna

from .fit_skew_t_pdf import fit_skew_t_pdf
from .plot.plot.plot_distribution import plot_distribution


def fit_skew_t_pdf_globally(feature_x_sample):
    """
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample, )
    Returns:
        int: n
        float: location
        float: scale
        float: df
        float: shape
    """

    values = feature_x_sample.values.flatten()

    values = values[~isna(values)]

    global_n, global_location, global_scale, global_degrees_of_freedom, global_shape = fit_skew_t_pdf(
        values)

    plot_distribution(
        values,
        distplot_kwargs={
            'norm_hist': True,
        },
        decorate_ax_kwargs={
            'title': 'Global',
        })

    print(
        'N={:.0f}   Location={:.2f}   Scale={:.2f}   Degrees of Freedom={:.2f}   Shape={:.2f}'.
        format(global_n, global_location, global_scale,
               global_degrees_of_freedom, global_shape))

    return global_n, global_location, global_scale, global_degrees_of_freedom, global_shape
