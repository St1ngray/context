from warnings import warn

from numpy import isnan, nanmean
from statsmodels.sandbox.distributions.extras import ACSkewT_gen


def fit_skew_t_pdf(array_1d,
                   skew_t_model=None,
                   fit_fixed_location=None,
                   fit_fixed_scale=None,
                   fit_initial_location=None,
                   fit_initial_scale=None):

    array_1d = array_1d.copy()
    is_nan = isnan(array_1d)
    if is_nan.all():
        raise ValueError('array_1d has only nan.')
    elif is_nan.any():
        warn('Replacing nan with mean ...')
        array_1d[is_nan] = nanmean(array_1d)

    if skew_t_model is None:
        skew_t_model = ACSkewT_gen()

    kwargs = {}

    if fit_fixed_location is not None:
        kwargs['floc'] = fit_fixed_location

    if fit_fixed_scale is not None:
        kwargs['fscale'] = fit_fixed_scale

    if fit_initial_location is not None:
        kwargs['loc'] = fit_initial_location

    if fit_initial_scale is not None:
        kwargs['scale'] = fit_initial_scale

    degree_of_freedom, shape, location, scale = skew_t_model.fit(
        array_1d, **kwargs)

    return array_1d.size, location, scale, degree_of_freedom, shape
