from numpy import ndarray
from statsmodels.sandbox.distributions.extras import ACSkewT_gen


def fit_skew_t_pdf(_1d_array,
                   skew_t_model=None,
                   fit_fixed_location=None,
                   fit_fixed_scale=None,
                   fit_initial_location=None,
                   fit_initial_scale=None):

    if not isinstance(_1d_array, ndarray):
        raise TypeError()

    if skew_t_model is None:

        skew_t_model = ACSkewT_gen()

    kwargs = {}

    if fit_fixed_location is not None:

        kwargs['floc'] = fit_fixed_location

    if fit_fixed_scale is not None:

        kwargs['fscale'] = fit_fixed_scale

    if fit_initial_location is not None:

        kwargs['loc'] = fit_initial_location

    else:

        kwargs['loc'] = _1d_array.mean()

    if fit_initial_scale is not None:

        kwargs['scale'] = fit_initial_scale

    else:

        kwargs['scale'] = _1d_array.std()

    degree_of_freedom, shape, location, scale = skew_t_model.fit(
        _1d_array, **kwargs)

    return _1d_array.size, location, scale, degree_of_freedom, shape
