from numpy import argmin, cumsum, linspace, where
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
        array: (n_grid); context indices
    """

    if not skew_t_model:
        skew_t_model = ACSkewT_gen()

    if any([p is None for p in [location, scale, df, shape]]):
        # Fit skew-t PDF
        location, scale, df, shape = fit_1d_array_to_skew_t_pdf(
            array_1d, skew_t_model=skew_t_model)

    # Compute PDF and PDF reflection
    grids = linspace(array_1d.min(), array_1d.max(), n_grid)
    pdf = skew_t_model.pdf(grids, df, shape, loc=location, scale=scale)
    pdf_reflection = skew_t_model.pdf(
        get_coordinates_for_reflection(grids, pdf),
        df,
        shape,
        loc=location,
        scale=scale)

    # Compute CDF and CDF reflection
    d = grids[1] - grids[0]
    d_area = pdf / pdf.sum() * d
    d_area_reflection = pdf_reflection / pdf_reflection.sum() * d
    if shape < 0:
        cdf = cumsum(d_area)
        cdf_reflection = cumsum(d_area_reflection)
    else:
        cdf = cumsum(d_area[::-1])[::-1]
        cdf_reflection = cumsum(d_area_reflection[::-1])[::-1]

    f0 = cdf
    f1 = cdf_reflection
    if shape < 0:
        context_indices = where(f1 < f0, ((f1 - f0) / f0), ((f1 - f0) / f1))
    else:
        context_indices = where(f1 < f0, ((f0 - f1) / f0), ((f0 - f1) / f1))

    context_indices = context_indices[[
        argmin(abs(grids - v)) for v in array_1d
    ]]

    return context_indices
