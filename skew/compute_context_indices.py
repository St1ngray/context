from numpy import argmax, argmin, linspace, log2, sqrt, where
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from summarize_context_indices import summarize_context_indices

from .fit_1d_array_to_skew_t_pdf import fit_1d_array_to_skew_t_pdf
from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection


def compute_context_indices(
        array_1d,
        skew_t_model=None,
        n_grid=3000,
        location=None,
        scale=None,
        df=None,
        shape=None,
        compute_context_indices_method='tail_reduction_reflection',
        degrees_of_freedom_for_tail_reduction=10e8):
    """
    Compute context indices.
    Arguments:
        array_1d (array): (n)
        skew_t_model (statsmodels.sandbox.distributions.extras.ACSkewT_gen):
        n_grid (int):
        location (float):
        scale (float):
        df (float):
        shape (float):
        compute_context_indices_method (str): 'tail_reduction' | 'reflection' |
            'tail_reduction_reflection'
        degrees_of_freedom_for_tail_reduction (number):
    Returns:
        dict: {
            fit: [n, location, scale, df, shape] (5),
            grid: array (n_grid),
            pdf: array (n_grid),
            pdf_transformed: array (n_grid),
            context_indices: array (n_grid),
            context_indices_like_array: array (n),
            context_summary: float,
        }
    """

    if skew_t_model is None:
        skew_t_model = ACSkewT_gen()

    if any([p is None for p in [location, scale, df, shape]]):
        # Fit skew-t PDF
        n, location, scale, df, shape = fit_1d_array_to_skew_t_pdf(
            array_1d, skew_t_model=skew_t_model)
    else:
        n = array_1d.size

    # Compute PDF
    grid = linspace(array_1d.min(), array_1d.max(), n_grid)
    pdf = skew_t_model.pdf(grid, df, shape, loc=location, scale=scale)

    # Compute context indices magnitude
    if compute_context_indices_method == 'tail_reduction':

        pdf_transformed = skew_t_model.pdf(
            grid,
            degrees_of_freedom_for_tail_reduction,
            shape,
            loc=location,
            scale=scale)

        context_indices_magnitude = where(pdf_transformed < pdf,
                                          ((pdf - pdf_transformed) / pdf), 0)

    elif compute_context_indices_method == 'reflection':

        pdf_transformed = skew_t_model.pdf(
            get_coordinates_for_reflection(grid, pdf),
            df,
            shape,
            loc=location,
            scale=scale)

        context_indices_magnitude = where(
            pdf_transformed < pdf, ((pdf - pdf_transformed) / pdf),
            ((pdf_transformed - pdf) / pdf_transformed))

    elif compute_context_indices_method == 'tail_reduction_reflection':

        pdf_transformed = skew_t_model.pdf(
            get_coordinates_for_reflection(grid, pdf),
            degrees_of_freedom_for_tail_reduction,
            shape,
            loc=location,
            scale=scale)

        context_indices_magnitude = where(pdf_transformed < pdf,
                                          ((pdf - pdf_transformed) / pdf), 0)

    else:
        raise ValueError('Unknown compute_context_indices_method {}.'.format(
            compute_context_indices_method))

    # Compute context indices
    pdf_argmax = argmax(pdf)
    signs = [-1] * pdf_argmax + [1] * (n_grid - pdf_argmax)
    context_indices = signs * context_indices_magnitude**(
        log2(df) / sqrt(abs(shape) * scale))

    # Make context_indices like array_1d
    context_indices_like_array = context_indices[[
        argmin(abs(grid - v)) for v in array_1d
    ]]

    # Summarize context indices
    context_summary = summarize_context_indices(context_indices_like_array,
                                                shape)

    return {
        'fit': [n, location, scale, df, shape],
        'grid': grid,
        'pdf': pdf,
        'pdf_transformed': pdf_transformed,
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array,
        'context_summary': context_summary,
    }
