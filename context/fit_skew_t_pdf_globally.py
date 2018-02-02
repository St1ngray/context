from pandas import isna

from .fit_skew_t_pdf import fit_skew_t_pdf
from .plot.plot.plot_distribution import plot_distribution


def fit_skew_t_pdf_globally(feature_x_sample,
                            fit_fixed_location=None,
                            fit_fixed_scale=None,
                            fit_initial_location=None,
                            fit_initial_scale=None):
    """
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample, )
        fit_fixed_location (float):
        fit_fixed_scale (float):
        fit_initial_location (float):
        fit_initial_scale (float):
    Returns:
        int: n
        float: location
        float: scale
        float: degree of freedom
        float: shape
    """

    values = feature_x_sample.values.flatten()

    is_missing = isna(values)
    print('Dropping {} missing values (of total {}) ...'.format(
        is_missing.sum(), is_missing.size))
    values = values[~is_missing]

    n, location, scale, degree_of_freedom, shape = fit_skew_t_pdf(
        values,
        fit_fixed_location=fit_fixed_location,
        fit_fixed_scale=fit_fixed_scale,
        fit_initial_location=fit_initial_location,
        fit_initial_scale=fit_initial_scale)

    plot_distribution(
        values,
        distplot_kwargs={
            'norm_hist': True,
        },
        decorate_ax_kwargs={
            'title': 'Global',
        })

    print(
        'N={:.0f}   Location={:.2f}   Scale={:.2f}   Degree of Freedom={:.2f}   Shape={:.2f}'.
        format(n, location, scale, degree_of_freedom, shape))

    return n, location, scale, degree_of_freedom, shape
