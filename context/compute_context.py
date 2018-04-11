from warnings import warn

from numpy import (absolute, asarray, concatenate, cumsum, finfo, isnan,
                   linspace, log, minimum, nanmean)
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_skew_t_pdf import fit_skew_t_pdf
from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection

EPS = finfo(float).eps


def compute_context(array_1d,
                    skew_t_model=None,
                    location=None,
                    scale=None,
                    degree_of_freedom=None,
                    shape=None,
                    fit_fixed_location=None,
                    fit_fixed_scale=None,
                    fit_initial_location=None,
                    fit_initial_scale=None,
                    n_grid=3000,
                    degree_of_freedom_for_tail_reduction=10e12,
                    global_location=None,
                    global_scale=None):

    array_1d = array_1d.copy()
    is_nan = isnan(array_1d)
    if is_nan.all():
        raise ValueError('array_1d has only nan.')
    elif is_nan.any():
        warn('Replacing nan with mean ...')
        array_1d[is_nan] = nanmean(array_1d)

    if skew_t_model is None:
        skew_t_model = ACSkewT_gen()

    if any(
            parameter is None
            for parameter in (location, scale, degree_of_freedom, shape)):

        n, location, scale, degree_of_freedom, shape = fit_skew_t_pdf(
            array_1d,
            skew_t_model=skew_t_model,
            fit_fixed_location=fit_fixed_location,
            fit_fixed_scale=fit_fixed_scale,
            fit_initial_location=fit_initial_location,
            fit_initial_scale=fit_initial_scale)
    else:
        n = array_1d.size

    grid = linspace(array_1d.min(), array_1d.max(), n_grid)

    pdf = skew_t_model.pdf(
        grid, degree_of_freedom, shape, loc=location, scale=scale)

    shape_pdf_reference = minimum(pdf,
                                  skew_t_model.pdf(
                                      get_coordinates_for_reflection(
                                          grid, pdf),
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

    shape_context_indices *= absolute(
        grid - grid[shape_pdf_reference_argmax]
    )  # * absolute(shape) / log(degree_of_freedom)

    if all(
            parameter is not None
            for parameter in (global_location, global_scale)):

        location_pdf_reference = minimum(pdf,
                                         skew_t_model.pdf(
                                             grid,
                                             degree_of_freedom,
                                             shape,
                                             loc=global_location,
                                             scale=scale))
        location_pdf_reference[location_pdf_reference < EPS] = EPS

        location_kl = pdf * log(pdf / location_pdf_reference)

        location_kl_darea = location_kl / location_kl.sum()
        location_pdf_reference_argmax = location_pdf_reference.argmax()
        location_context_indices = concatenate(
            (-cumsum(
                location_kl_darea[:location_pdf_reference_argmax][::-1])[::-1],
             cumsum(location_kl_darea[location_pdf_reference_argmax:])))

        location_context_indices *= absolute(
            grid - grid[location_pdf_reference_argmax]) / (
                scale + global_scale)

        context_indices = location_context_indices + shape_context_indices

    else:
        location_pdf_reference = None
        location_context_indices = None
        context_indices = shape_context_indices

    context_indices_like_array = context_indices[[
        absolute(grid - value).argmin() for value in array_1d
    ]]

    negative_context_summary = context_indices_like_array[
        context_indices_like_array < 0].sum()

    positive_context_summary = context_indices_like_array[
        0 < context_indices_like_array].sum()

    return {
        'fit': asarray((n, location, scale, degree_of_freedom, shape)),
        'grid': grid,
        'pdf': pdf,
        'shape_pdf_reference': shape_pdf_reference,
        'shape_context_indices': shape_context_indices,
        'location_pdf_reference': location_pdf_reference,
        'location_context_indices': location_context_indices,
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array,
        'negative_context_summary': negative_context_summary,
        'positive_context_summary': positive_context_summary
    }
