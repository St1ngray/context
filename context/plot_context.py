from numpy import absolute, isnan, nanmean
from plotly.graph_objs import Histogram, Scatter

from .compute_context import compute_context
from .plot.plot.plot_and_save import plot_and_save


def plot_context(array_1d,
                 title,
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
                 y_max_is_pdf_max=True,
                 xaxis_title='Value',
                 html_file_path=None):
    """
    Plot context.
    Arguments:
        array_1d (ndarray): (n, )
        title (str):
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
        y_max_is_pdf_max (bool):
        xaxis_title (str):
        html_file_path (str):
    Returns:
    """

    array_1d = array_1d.copy()
    is_nan = isnan(array_1d)
    if is_nan.all():
        raise ValueError('array_1d has only nan.')
    else:
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
    context_indices_absolute_max = absolute(
        context_dict['context_indices']).max()
    if y_max_is_pdf_max:
        y_max = pdf_max
    else:
        y_max = max(pdf_max, context_indices_absolute_max)

    data = []

    data.append(
        Histogram(
            name='Histogram', x=array_1d, histnorm='probability density'))
    data.append(
        Scatter(name='PDF', x=context_dict['grid'], y=context_dict['pdf']))
    data.append(
        Scatter(
            name='Reflection PDF Reference',
            x=context_dict['grid'],
            y=context_dict['r_pdf_reference']))
    if context_dict['s_pdf_reference'] is not None:
        data.append(
            Scatter(
                name='Shift PDF Reference',
                x=context_dict['grid'],
                y=context_dict['s_pdf_reference']))

    y = absolute(context_dict['context_indices'])
    if y_max_is_pdf_max and y_max < context_indices_absolute_max:
        y /= context_indices_absolute_max
        y *= y_max
    data.append(Scatter(name='Context Indices', x=context_dict['grid'], y=y))

    positive_context_indices = 0 <= context_dict['context_indices']

    x = context_dict['grid'][~positive_context_indices]
    y0 = -context_dict['context_indices'][~positive_context_indices]
    y1 = absolute(context_dict['r_context_indices'])[~positive_context_indices]
    if y_max_is_pdf_max and y_max < context_indices_absolute_max:
        if y0.size:
            y0 /= context_indices_absolute_max
            y0 *= y_max
        if y1.size:
            y1 /= context_indices_absolute_max
            y1 *= y_max
    data.append(
        Scatter(
            name='- Reflection Context Indices', x=x, y=y1, fill='tozeroy'))
    data.append(Scatter(name='- Context Indices', x=x, y=y0, fill='tonexty'))

    x = context_dict['grid'][positive_context_indices]
    y0 = context_dict['context_indices'][positive_context_indices]
    y1 = context_dict['r_context_indices'][positive_context_indices]
    if y_max_is_pdf_max and y_max < context_indices_absolute_max:
        if y0.size:
            y0 /= context_indices_absolute_max
            y0 *= y_max
        if y1.size:
            y1 /= context_indices_absolute_max
            y1 *= y_max
    data.append(
        Scatter(
            name='+ Reflection Context Indices', x=x, y=y1, fill='tozeroy'))
    data.append(Scatter(name='+ Context Indices', x=x, y=y0, fill='tonexty'))

    layout = dict(title=title, xaxis=dict(title=xaxis_title))

    figure = dict(data=data, layout=layout)

    plot_and_save(figure, html_file_path)
