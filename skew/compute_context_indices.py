from numpy import argmin, cumsum, linspace, sign, where
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_1d_array_to_skew_t_pdf import fit_1d_array_to_skew_t_pdf
from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection


def compute_context_indices(array_1d,
                            skew_t_model=None,
                            n_grid=3000,
                            location=None,
                            scale=None,
                            df=None,
                            shape=None):
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
    Returns:
        dict: {
            fit: [n, location, scale, df, shape] (5),
            grid: array (n_grid),
            pdf: array (n),
            pdf_reflection: array (n),
            cdf: array (n),
            cdf_reflection: array (n),
            context_indices: array (n),
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

    # Compute PDF and PDF reflection
    grid = linspace(array_1d.min(), array_1d.max(), n_grid)
    pdf = skew_t_model.pdf(grid, df, shape, loc=location, scale=scale)
    pdf_reflection = skew_t_model.pdf(
        get_coordinates_for_reflection(grid, pdf),
        df,
        shape,
        loc=location,
        scale=scale)

    # Compute CDF and CDF reflection
    if shape < 0:
        cdf = cumsum(pdf / pdf.sum())
        cdf_reflection = cumsum(pdf_reflection / pdf_reflection.sum())
    else:
        cdf = cumsum(pdf[::-1] / pdf.sum())[::-1]
        cdf_reflection = cumsum(
            pdf_reflection[::-1] / pdf_reflection.sum())[::-1]

    # Compute context indices
    f0 = pdf
    f1 = pdf_reflection
    context_indices = where(f1 < f0, ((f0 - f1) / f0), ((f0 - f1) / f1))
    context_indices = sign(context_indices) * abs(context_indices)**(
        df / scale) * sign(shape)

    context_indices_like_array = context_indices[[
        argmin(abs(grid - v)) for v in array_1d
    ]]

    context_summary = ((sign(context_indices_like_array) == sign(shape)) *
                       context_indices_like_array).sum()

    return {
        'fit': [n, location, scale, df, shape],
        'grid': grid,
        'pdf': pdf,
        'pdf_reflection': pdf_reflection,
        'cdf': cdf,
        'cdf_reflection': cdf_reflection,
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array,
        'context_summary': context_summary,
    }
