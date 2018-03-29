from warnings import warn

from numpy import absolute, isnan, nanmean
from pandas import Series

from .compute_context import compute_context
from .plot.plot.plot_and_save import plot_and_save
from .plot.plot.plot_distributions import plot_distributions


def plot_context(array_1d,
                 text=None,
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
                 global_scale=None,
                 title='Context Plot',
                 xaxis_title='Value',
                 y_max_is_pdf_max=True,
                 line_width=3.2,
                 html_file_path=None):
    """
    Plot context.
    Arguments:
        array_1d (ndarray | Series): (n)
        text (iterable): (n)
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
        title (str):
        xaxis_title (str):
        y_max_is_pdf_max (bool):
        line_width (float):
        html_file_path (str):
    Returns:
    """

    if isinstance(array_1d, Series):
        if title is None:
            title = array_1d.name
        if text is None:
            text = array_1d.index
        array_1d = array_1d.values

    array_1d = array_1d.copy()
    is_nan = isnan(array_1d)
    if is_nan.all():
        raise ValueError('array_1d has only nan.')
    elif is_nan.any():
        warn('Replacing nan with mean ...')
        array_1d[is_nan] = nanmean(array_1d)

    context_dict = compute_context(
        array_1d,
        location=location,
        scale=scale,
        degree_of_freedom=degree_of_freedom,
        shape=shape,
        fit_fixed_location=fit_fixed_location,
        fit_fixed_scale=fit_fixed_scale,
        fit_initial_location=fit_initial_location,
        fit_initial_scale=fit_initial_scale,
        n_grid=n_grid,
        degree_of_freedom_for_tail_reduction=
        degree_of_freedom_for_tail_reduction,
        global_location=global_location,
        global_scale=global_scale)

    pdf_max = context_dict['pdf'].max()

    absolute_context_indices = absolute(context_dict['context_indices'])
    absolute_context_indices_max = absolute_context_indices.max()

    if y_max_is_pdf_max:
        y_max = pdf_max
        if y_max < absolute_context_indices_max:
            absolute_context_indices = absolute_context_indices / absolute_context_indices_max * y_max
    else:
        y_max = max(pdf_max, absolute_context_indices_max)

    figure = plot_distributions(
        xs=(array_1d, ),
        texts=(text, ),
        names=('Distribution', ),
        title=title,
        xaxis_title=xaxis_title,
        plot=False)

    grid = context_dict['grid']

    pdf = context_dict['pdf']
    figure['data'].append(
        dict(
            type='scatter',
            name='PDF',
            x=grid,
            y=pdf,
            line=dict(width=line_width, color='#20d9ba')))

    shape_pdf_reference = context_dict['shape_pdf_reference']
    shape_pdf_reference[pdf <= shape_pdf_reference] = None
    figure['data'].append(
        dict(
            type='scatter',
            name='Shape PDF Reference',
            x=grid,
            y=shape_pdf_reference,
            line=dict(width=line_width, color='#9017e6')))

    location_pdf_reference = context_dict['location_pdf_reference']
    if location_pdf_reference is not None:
        location_pdf_reference[pdf <= location_pdf_reference] = None
        figure['data'].append(
            dict(
                type='scatter',
                name='Location PDF Reference',
                x=grid,
                y=location_pdf_reference,
                line=dict(width=line_width, color='#4e40d8')))

    negative_context_indices = context_dict['context_indices'] < 0
    for name, indices, color in (('- Context Indices',
                                  negative_context_indices, '#0088ff'),
                                 ('+ Context Indices',
                                  ~negative_context_indices, '#ff1968')):
        figure['data'].append(
            dict(
                type='scatter',
                name=name,
                x=grid[indices],
                y=absolute_context_indices[indices],
                fill='tozeroy',
                line=dict(width=line_width, color=color)))

    figure['layout'].update(title=title, width=1000)

    plot_and_save(figure, html_file_path)
