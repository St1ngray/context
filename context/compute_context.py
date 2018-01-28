from numpy import (absolute, argmax, argmin, array, convolve, linspace, log,
                   sign, sqrt, where)
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_skew_t_pdf import fit_skew_t_pdf
from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection


def compute_context(array_1d,
                    skew_t_model=None,
                    location=None,
                    scale=None,
                    df=None,
                    shape=None,
                    fit_fixed_location=None,
                    fit_fixed_scale=None,
                    n_grid=3000,
                    true_mean=None,
                    degrees_of_freedom_for_tail_reduction=10e8):
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
        true_mean (float):
        degrees_of_freedom_for_tail_reduction (float):
    Returns:
        dict: {
            fit: ndarray; (5, ) (N, Location, Scale, DF, Shape),
            grid: ndarray; (n_grid, ),
            pdf: ndarray; (n_grid, ),
            pdf_reference: ndarray; (n_grid, ),
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

    grid_for_reflection = argmax(pdf_reference)

    if true_mean is not None:

        location_penalties = absolute((grid - true_mean) / location)
        pdf_reference = pdf_reference**location_penalties

    context_indices_magnitude = where(
        pdf_reference < pdf, ((pdf - pdf_reference) / pdf),
        0)**(log(df) / sqrt(absolute(shape) * scale))

    context_indices = context_indices_magnitude * (
        (-1, ) * grid_for_reflection + (1, ) * (n_grid - grid_for_reflection))

    context_indices_like_array = context_indices[[
        argmin(absolute(grid - value)) for value in array_1d
    ]]

    absolute_value_weighted_context_like_array = absolute(
        array_1d) * context_indices_like_array

    negative_context_summary = absolute_value_weighted_context_like_array[:
                                                                          grid_for_reflection].sum(
                                                                          )
    positive_context_summary = absolute_value_weighted_context_like_array[
        grid_for_reflection:].sum()

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
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array,
        'context_summary': context_summary,
    }
