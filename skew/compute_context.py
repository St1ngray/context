from numpy import absolute, argmax, argmin, linspace, log, sign, sqrt, where
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
                    n_grid=3000,
                    compute_context_method='tail_reduction_reflection',
                    degrees_of_freedom_for_tail_reduction=10e8,
                    summarize_context_by='absolute_value_weighted_context',
                    summarize_context_side='shape_side'):
    """
    Compute context.
    Arguments:
        array_1d (ndarray): (n, )
        skew_t_model (statsmodels.sandbox.distributions.extras.ACSkewT_gen):
        location (float):
        scale (float):
        df (float):
        shape (float):
        n_grid (int):
        compute_context_method (str): 'tail_reduction_reflection' |
            'tail_reduction' | 'reflection' |
        degrees_of_freedom_for_tail_reduction (float):
        summarize_context_by (str): 'absolute_value_weighted_context' |
            'context'
        summarize_context_side (str): 'shape_side' | 'both_sides'
    Returns:
        dict: {
            fit: [n, location, scale, df, shape] (5, ),
            grid: array (n_grid, ),
            pdf: array (n_grid, ),
            pdf_transformed: array (n_grid, ),
            context_indices: array (n_grid, ),
            context_indices_like_array: array (n, ),
            conext_summary: float,
        }
    """

    if skew_t_model is None:
        skew_t_model = ACSkewT_gen()

    if any([p is None for p in [location, scale, df, shape]]):
        n, location, scale, df, shape = fit_skew_t_pdf(
            array_1d, skew_t_model=skew_t_model)
    else:
        n = array_1d.size

    grid = linspace(array_1d.min(), array_1d.max(), n_grid)

    pdf = skew_t_model.pdf(grid, df, shape, loc=location, scale=scale)

    if compute_context_method == 'reflection':

        pdf_transformed = skew_t_model.pdf(
            get_coordinates_for_reflection(grid, pdf),
            df,
            shape,
            loc=location,
            scale=scale)

        context_indices_magnitude = where(
            pdf_transformed < pdf, ((pdf - pdf_transformed) / pdf),
            ((pdf_transformed - pdf) / pdf_transformed))

    elif compute_context_method == 'tail_reduction':

        pdf_transformed = skew_t_model.pdf(
            grid,
            degrees_of_freedom_for_tail_reduction,
            shape,
            loc=location,
            scale=scale)

        context_indices_magnitude = where(pdf_transformed < pdf,
                                          ((pdf - pdf_transformed) / pdf), 0)

    elif compute_context_method == 'tail_reduction_reflection':

        pdf_transformed = skew_t_model.pdf(
            get_coordinates_for_reflection(grid, pdf),
            degrees_of_freedom_for_tail_reduction,
            shape,
            loc=location,
            scale=scale)

        context_indices_magnitude = where(pdf_transformed < pdf,
                                          ((pdf - pdf_transformed) / pdf), 0)

    else:
        raise ValueError('Unknown compute_context_method: {}.'.format(
            compute_context_method))

    pdf_argmax = argmax(pdf)
    context_indices = ([-1] * pdf_argmax + [1] *
                       (n_grid - pdf_argmax)) * context_indices_magnitude**(
                           log(df) / sqrt(abs(shape) * scale))

    context_indices_like_array = context_indices[[
        argmin(abs(grid - v)) for v in array_1d
    ]]

    if summarize_context_by == 'context':
        a = context_indices_like_array

    elif summarize_context_by == 'absolute_value_weighted_context':
        a = absolute(array_1d) * context_indices_like_array

    else:
        raise ValueError(
            'Unknown summarize_context_by: {}.'.format(summarize_context_by))

    if summarize_context_side == 'shape_side':
        context_summary = ((sign(a) == sign(shape)) * a).sum()

    elif summarize_context_side == 'both_sides':
        context_summary = a.sum()

    else:
        raise ValueError('Unknown summarize_context_side: {}.'.format(
            summarize_context_side))

    return {
        'fit': [n, location, scale, df, shape],
        'grid': grid,
        'pdf': pdf,
        'pdf_transformed': pdf_transformed,
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array,
        'context_summary': context_summary,
    }
