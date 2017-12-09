from os.path import join

from matplotlib.pyplot import close, figure, gca, plot, show
from numpy import argmin, asarray, cumsum, empty, linspace, log, sign, where
from pandas import DataFrame
from seaborn import distplot
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection
from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .plot.plot.decorate import decorate
from .plot.plot.save_plot import save_plot
from .plot.plot.style import CMAP_CATEGORICAL_TAB20, FIGURE_SIZE


def make_essentiality_matrix(feature_x_sample,
                             feature_x_fit,
                             n_grids=3000,
                             function='scaled_fractional_difference',
                             factor=1):
    """

    Arguments:
        feature_x_sample: DataFrame; (n_feature, n_sample)
        feature_x_fit: DataFrame;
        n_grids: int;
        function: str;
        factor: number;
    Returns:
        DataFrame; (n_feature, n_sample)
    """

    print('\tApplying {} to each feature ...'.format(function))

    empty_ = empty(feature_x_sample.shape)

    skew_t = ACSkewT_gen()

    for i, (f_i, f_v) in enumerate(feature_x_sample.iterrows()):

        # Build skew-t PDF
        grids = linspace(f_v.min(), f_v.max(), n_grids)
        n, df, shape, location, scale = feature_x_fit.ix[i, :]
        skew_t_pdf = skew_t.pdf(grids, df, shape, loc=location, scale=scale)

        # Build reflected skew-t PDF
        skew_t_pdf_r = skew_t.pdf(
            get_coordinates_for_reflection(grids, skew_t_pdf),
            df,
            shape,
            loc=location,
            scale=scale)

        # Set up function
        if function.startswith('scaled_fractional_difference'):
            function = 'where(f2 < f1, ((f1 - f2) / f1)**{}, 0)'.format(scale)

        ei = _compute_essentiality_index(skew_t_pdf, skew_t_pdf_r, function,
                                         ['+',
                                          '-'][shape > 0], grids[1] - grids[0])

        ei = normalize_1d_array(ei, '0-1')

        empty_[i, :] = ei[[argmin(abs(grids - x))
                           for x in asarray(f_v)]] * sign(shape) * factor

    return DataFrame(
        empty_, index=feature_x_sample.index, columns=feature_x_sample.columns)
