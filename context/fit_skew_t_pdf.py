from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .process_array_1d_bad_values import process_array_1d_bad_values


def fit_skew_t_pdf(_1d_array,
                   skew_t_model=None,
                   fit_fixed_location=None,
                   fit_fixed_scale=None,
                   fit_initial_location=None,
                   fit_initial_scale=None):

    _1d_array = process_array_1d_bad_values(_1d_array)

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
        _1d_array, **kwargs)

    return _1d_array.size, location, scale, degree_of_freedom, shape
