from warnings import warn

from numpy import absolute, isnan, nanmean
from pandas import Series

from .compute_context import compute_context
from .plot.plot.plot_and_save import plot_and_save


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
                 html_file_path=None):

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

    layout = dict(
        width=800,
        height=800,
        title=title,
        xaxis1=dict(title=xaxis_title, anchor='y1'),
        yaxis1=dict(
            domain=(0, 0.16), dtick=1, showticklabels=False, zeroline=False),
        yaxis2=dict(domain=(0.24, 1), zeroline=False),
        barmode='overlay',
        legend=dict(orientation='h'))

    data = []

    data.append(
        dict(
            type='histogram',
            name='Data',
            legendgroup='Data',
            yaxis='y2',
            x=array_1d,
            marker=dict(color='#20d9ba'),
            histnorm='probability density',
            hoverinfo='x+y'))

    data.append(
        dict(
            type='scatter',
            legendgroup='Data',
            showlegend=False,
            x=array_1d,
            y=(0, ) * array_1d.size,
            text=text,
            mode='markers',
            marker=dict(symbol='line-ns-open', color='#20d9ba'),
            hoverinfo='x+text'))

    grid = context_dict['grid']
    line_width = 3.2

    pdf = context_dict['pdf']
    data.append(
        dict(
            type='scatter',
            yaxis='y2',
            name='PDF',
            x=grid,
            y=pdf,
            line=dict(width=line_width, color='#24e7c0')))

    shape_pdf_reference = context_dict['shape_pdf_reference']
    shape_pdf_reference[pdf <= shape_pdf_reference] = None
    data.append(
        dict(
            type='scatter',
            yaxis='y2',
            name='Shape Reference',
            x=grid,
            y=shape_pdf_reference,
            line=dict(width=line_width, color='#9017e6')))

    location_pdf_reference = context_dict['location_pdf_reference']
    if location_pdf_reference is not None:
        layout['legend'].update(tracegroupgap=3)

        location_pdf_reference[pdf <= location_pdf_reference] = None
        data.append(
            dict(
                type='scatter',
                yaxis='y2',
                name='Location Reference',
                x=grid,
                y=location_pdf_reference,
                line=dict(width=line_width, color='#4e40d8')))

    is_negative = context_dict['context_indices'] < 0
    for name, indices, color in (('- Context', is_negative, '#0088ff'),
                                 ('+ Context', ~is_negative, '#ff1968')):
        data.append(
            dict(
                type='scatter',
                yaxis='y2',
                name=name,
                x=grid[indices],
                y=absolute_context_indices[indices],
                line=dict(width=line_width, color=color),
                fill='tozeroy'))

    plot_and_save(dict(layout=layout, data=data), html_file_path)
