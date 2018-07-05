from numpy import (absolute, asarray, concatenate, cumsum, finfo, full,
                   linspace, log, minimum, nan)
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_skew_t_pdf import fit_skew_t_pdf
from .nd_array.nd_array.check_nd_array_for_bad_value import \
    check_nd_array_for_bad_value
from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection

EPS = finfo(float).eps


def compute_context(_1d_array,
                    skew_t_model=None,
                    location=None,
                    scale=None,
                    degree_of_freedom=None,
                    shape=None,
                    fit_fixed_location=None,
                    fit_fixed_scale=None,
                    fit_initial_location=None,
                    fit_initial_scale=None,
                    n_grid=1e3,
                    degree_of_freedom_for_tail_reduction=1e8,
                    multiply_distance_from_location=False,
                    global_location=None,
                    global_scale=None,
                    global_degree_of_freedom=None,
                    global_shape=None):

    is_bad_value = check_nd_array_for_bad_value(
        _1d_array, raise_for_bad_value=False)

    _1d_array_good = _1d_array[~is_bad_value]

    if skew_t_model is None:

        skew_t_model = ACSkewT_gen()

    if any(
            parameter is None
            for parameter in (location, scale, degree_of_freedom, shape)):

        n, location, scale, degree_of_freedom, shape = fit_skew_t_pdf(
            _1d_array_good,
            skew_t_model=skew_t_model,
            fit_fixed_location=fit_fixed_location,
            fit_fixed_scale=fit_fixed_scale,
            fit_initial_location=fit_initial_location,
            fit_initial_scale=fit_initial_scale)

    else:

        n = _1d_array_good.size

    grid = linspace(_1d_array_good.min(), _1d_array_good.max(), n_grid)

    pdf = skew_t_model.pdf(
        grid, degree_of_freedom, shape, loc=location, scale=scale)

    shape_pdf_reference = minimum(
        pdf,
        skew_t_model.pdf(
            get_coordinates_for_reflection(grid, pdf),
            degree_of_freedom_for_tail_reduction,
            shape,
            loc=location,
            scale=scale))

    shape_pdf_reference[shape_pdf_reference < EPS] = EPS

    shape_kl = pdf * log(pdf / shape_pdf_reference)

    shape_kl_darea = shape_kl / shape_kl.sum()

    shape_pdf_reference_argmax = shape_pdf_reference.argmax()

    shape_context_indices = concatenate(
        (-cumsum(shape_kl_darea[:shape_pdf_reference_argmax][::-1])[::-1],
         cumsum(shape_kl_darea[shape_pdf_reference_argmax:])))

    if multiply_distance_from_location:

        shape_context_indices *= absolute(grid -
                                          grid[shape_pdf_reference_argmax])

    shape_context_indices *= (1 + absolute(shape)) / (
        scale * log(1 + degree_of_freedom))

    if all(
            parameter is not None
            for parameter in (global_location, global_scale,
                              global_degree_of_freedom, global_shape)):

        location_pdf_reference = minimum(
            pdf,
            skew_t_model.pdf(
                grid,
                global_degree_of_freedom,
                global_shape,
                loc=global_location,
                scale=global_scale))

        location_pdf_reference[location_pdf_reference < EPS] = EPS

        location_kl = pdf * log(pdf / location_pdf_reference)

        location_kl_darea = location_kl / location_kl.sum()

        location_pdf_reference_argmax = location_pdf_reference.argmax()

        location_context_indices = concatenate(
            (-cumsum(
                location_kl_darea[:location_pdf_reference_argmax][::-1])[::-1],
             cumsum(location_kl_darea[location_pdf_reference_argmax:])))

        location_context_indices *= absolute(
            grid - grid[location_pdf_reference_argmax])

        location_context_indices /= scale + global_scale

        context_indices = location_context_indices + shape_context_indices

    else:

        location_pdf_reference = None

        location_context_indices = None

        context_indices = shape_context_indices

    context_indices_like_array = full(_1d_array.size, nan)

    context_indices_like_array[~is_bad_value] = context_indices[[
        absolute(grid - value).argmin() for value in _1d_array_good
    ]]

    return {
        'fit': asarray((n, location, scale, degree_of_freedom, shape)),
        'grid': grid,
        'pdf': pdf,
        'shape_pdf_reference': shape_pdf_reference,
        'shape_context_indices': shape_context_indices,
        'location_pdf_reference': location_pdf_reference,
        'location_context_indices': location_context_indices,
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array
    }
