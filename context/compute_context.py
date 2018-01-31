from numpy import (absolute, argmax, argmin, array, linspace, log, sqrt, where,
                   zeros_like)
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_skew_t_pdf import fit_skew_t_pdf
from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection
from .nd_array.nd_array.get_intersections_between_2_1d_arrays import \
    get_intersections_between_2_1d_arrays


def compute_context(array_1d,
                    skew_t_model=None,
                    location=None,
                    scale=None,
                    df=None,
                    shape=None,
                    fit_fixed_location=None,
                    fit_fixed_scale=None,
                    n_grid=3000,
                    degrees_of_freedom_for_tail_reduction=10e8,
                    global_location=None,
                    global_scale=None,
                    global_shape=None):
    """
    Compute context.
    Arguments:
        array_1d (ndarray): (n, )
        skew_t_model (statsmodels.sandbox.distributions.extras.ACSkewT_gen):
        location (float):
        scale (float):
        df (float):
        shape (float):
        fit_fixed_location (float):
        fit_fixed_scale (float):
        n_grid (int):
        degrees_of_freedom_for_tail_reduction (float):
        global_location (float):
        global_scale (float):
        global_shape (float):
    Returns:
        dict: {
            fit: ndarray; (5, ) (N, Location, Scale, DF, Shape),
            grid: ndarray; (n_grid, ),
            pdf: ndarray; (n_grid, ),
            pdf_reference: ndarray; (n_grid, ),
            global_pdf_reference: ndarray; (n_grid, ),
            context_indices: ndarray; (n_grid, ),
            context_indices_like_array: ndarray; (n, ),
            context_summary: float,
        }
    """

    if skew_t_model is None:
        skew_t_model = ACSkewT_gen()

    if any((parameter is None for parameter in (
            location,
            scale,
            df,
            shape, ))):

        n, location, scale, df, shape = fit_skew_t_pdf(
            array_1d,
            skew_t_model=skew_t_model,
            fit_fixed_location=fit_fixed_location,
            fit_fixed_scale=fit_fixed_scale)
    else:
        n = array_1d.size

    grid = linspace(array_1d.min(), array_1d.max(), n_grid)

    pdf = skew_t_model.pdf(grid, df, shape, loc=location, scale=scale)

    pdf_reference = skew_t_model.pdf(
        get_coordinates_for_reflection(grid, pdf),
        degrees_of_freedom_for_tail_reduction,
        shape,
        loc=location,
        scale=scale)

    context_indices = (pdf - pdf_reference) / pdf
    context_indices[context_indices < 0] = 0

    penalty = log(df) / sqrt(absolute(shape) * scale)
    context_indices = context_indices**penalty

    i = argmax(pdf_reference)
    context_indices = ((-1, ) * i + (1, ) * (n_grid - i)) * context_indices

    if all(
            parameter is not None
            for parameter in (
                global_location,
                global_scale,
                global_shape, )):

        global_pdf_reference = skew_t_model.pdf(
            grid,
            degrees_of_freedom_for_tail_reduction,
            global_shape,
            loc=global_location,
            scale=global_scale)

        global_context_indices = (pdf - global_pdf_reference) / pdf

        penalty *= 1 + (scale / global_scale) / (
            absolute(location - global_location) / global_scale)

        global_context_indices = global_context_indices**penalty

        is_intersection = get_intersections_between_2_1d_arrays(
            pdf, global_pdf_reference)

        if location < global_location:
            global_context_indices *= grid < grid[is_intersection][0]
            global_context_indices *= -1
        else:
            global_context_indices *= grid[is_intersection][1] < grid

        context_indices = where(
            absolute(context_indices) < absolute(global_context_indices),
            global_context_indices, context_indices)

    else:
        global_pdf_reference = zeros_like(pdf_reference)

    context_indices_like_array = context_indices[[
        argmin(absolute(grid - value)) for value in array_1d
    ]]

    absolute_value_weighted_context_like_array = absolute(
        array_1d) * context_indices_like_array

    negative_context_summary = absolute_value_weighted_context_like_array[
        absolute_value_weighted_context_like_array < 0].sum()

    positive_context_summary = absolute_value_weighted_context_like_array[
        0 < absolute_value_weighted_context_like_array].sum()

    if absolute(negative_context_summary) < absolute(positive_context_summary):
        context_summary = positive_context_summary
    else:
        context_summary = negative_context_summary

    return {
        'fit': array((
            n,
            location,
            scale,
            df,
            shape, )),
        'grid': grid,
        'pdf': pdf,
        'pdf_reference': pdf_reference,
        'global_pdf_reference': global_pdf_reference,
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array,
        'context_summary': context_summary,
    }
