from numpy import (absolute, array, concatenate, cumsum, finfo, isnan,
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
                    degree_of_freedom_for_tail_reduction=10e8,
                    global_location=None,
                    global_scale=None):
    """
    Compute context.
    Arguments:
        array_1d (ndarray): (n, )
        skew_t_model (statsmodels.sandbox.distributions.extras.ACSkewT_gen):
        location (float):
        scale (float):
        degree_of_freedom (float):
        shape (float):
        fit_fixed_location (float):
        fit_fixed_scale (float):
        fit_initial_location (float):
        fit_initial_scale (float):
        n_grid (int):
        degree_of_freedom_for_tail_reduction (float):
        global_location (float):
        global_scale (float):
    Returns:
        dict: {
            fit: ndarray; (5, ) (N, Location, Scale, DF, Shape, ),
            grid: ndarray; (n_grid, ),
            pdf: ndarray; (n_grid, ),
            r_pdf_reference: ndarray; (n_grid, ),
            r_context_indices: ndarray; (n_grid, ),
            s_pdf_reference: ndarray; (n_grid, ),
            s_context_indices: ndarray; (n_grid, ),
            context_indices: ndarray; (n_grid, ),
            context_indices_like_array: ndarray; (n, ),
            context_summary: float,
        }
    """

    array_1d = array_1d.copy()
    is_nan = isnan(array_1d)
    if is_nan.all():
        raise ValueError('array_1d has only nan.')
    else:
        array_1d[is_nan] = nanmean(array_1d)

    if skew_t_model is None:
        skew_t_model = ACSkewT_gen()

    if any(
            parameter is None
            for parameter in (
                location,
                scale,
                degree_of_freedom,
                shape, )):

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

    r_pdf_reference = minimum(pdf,
                              skew_t_model.pdf(
                                  get_coordinates_for_reflection(grid, pdf),
                                  degree_of_freedom_for_tail_reduction,
                                  shape,
                                  loc=location,
                                  scale=scale))
    r_pdf_reference[r_pdf_reference < EPS] = EPS

    r_kl = pdf * log(pdf / r_pdf_reference)

    r_pdf_reference_argmax = r_pdf_reference.argmax()

    r_kl_darea = r_kl / r_kl.sum()
    r_context_indices = concatenate((
        -cumsum(r_kl_darea[:r_pdf_reference_argmax][::-1])[::-1],
        cumsum(r_kl_darea[r_pdf_reference_argmax:]), ))

    r_context_indices *= absolute(grid - grid[r_pdf_reference_argmax]
                                  ) * absolute(shape) / log(degree_of_freedom)

    if all(
            parameter is not None
            for parameter in (
                global_location,
                global_scale, )):

        s_pdf_reference = minimum(pdf,
                                  skew_t_model.pdf(
                                      grid,
                                      degree_of_freedom,
                                      shape,
                                      loc=global_location,
                                      scale=scale))
        s_pdf_reference[s_pdf_reference < EPS] = EPS

        s_kl = pdf * log(pdf / s_pdf_reference)

        s_pdf_reference_argmax = s_pdf_reference.argmax()

        s_kl_darea = s_kl / s_kl.sum()
        s_context_indices = concatenate((
            -cumsum(s_kl_darea[:s_pdf_reference_argmax][::-1])[::-1],
            cumsum(s_kl_darea[s_pdf_reference_argmax:]), ))

        s_context_indices *= absolute(grid - grid[s_pdf_reference_argmax]) / (
            scale + global_scale)

        context_indices = s_context_indices + r_context_indices

    else:
        s_pdf_reference = s_context_indices = None
        context_indices = r_context_indices

    context_indices_like_array = context_indices[[
        absolute(grid - value).argmin() for value in array_1d
    ]]

    negative_context_summary = context_indices_like_array[
        context_indices_like_array < 0].sum()

    positive_context_summary = context_indices_like_array[
        0 < context_indices_like_array].sum()

    return {
        'fit': array((
            n,
            location,
            scale,
            degree_of_freedom,
            shape, )),
        'grid': grid,
        'pdf': pdf,
        'r_pdf_reference': r_pdf_reference,
        'r_context_indices': r_context_indices,
        's_pdf_reference': s_pdf_reference,
        's_context_indices': s_context_indices,
        'context_indices': context_indices,
        'context_indices_like_array': context_indices_like_array,
        'negative_context_summary': negative_context_summary,
        'positive_context_summary': positive_context_summary,
    }
