from warnings import warn

from numpy import absolute
from pandas import Series

from .compute_context import compute_context
from .plot.plot.plot_and_save import plot_and_save


def plot_context(_1d_array,
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
                 y_max_is_pdf_max=True,
                 plot_rug=True,
                 title='Context Plot',
                 xaxis_title='',
                 html_file_path=None,
                 plotly_file_path=None):

    if isinstance(_1d_array, Series):

        if title is None:

            title = _1d_array.name

        if text is None:

            text = _1d_array.index

        _1d_array = _1d_array.values

    context_dict = compute_context(
        _1d_array,
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

    if 10000 < _1d_array.size:

        warn('Set plot_rug to False because 10000 < _1d_array.size.')

        plot_rug = False

    if plot_rug:

        yaxis_max = 0.16
        yaxis2_min = yaxis_max + 0.08

    else:

        yaxis_max = 0
        yaxis2_min = 0

    layout = dict(
        width=960,
        height=640,
        title=title,
        xaxis=dict(anchor='y', title=xaxis_title),
        yaxis=dict(
            domain=(0, yaxis_max),
            dtick=1,
            zeroline=False,
            showticklabels=False),
        yaxis2=dict(domain=(yaxis2_min, 1)),
        legend=dict(orientation='h'))

    annotations = []

    for i, (template, fit_parameter) in enumerate(
            zip(('N={:.0f}', 'Location={:.2f}', 'Scale={:.2f}', 'DF={:.2f}',
                 'Shape={:.2f}'), context_dict['fit'])):

        annotations.append(
            dict(
                xref='paper',
                yref='paper',
                x=(i + 1) / (5 + 1),
                y=1.051,
                xanchor='center',
                text=template.format(fit_parameter),
                font=dict(color='#ffffff'),
                bgcolor='#003171',
                bordercolor='#ebf6f7',
                borderpad=5.1,
                showarrow=False))

    layout.update(annotations=annotations)

    data = []

    data.append(
        dict(
            yaxis='y2',
            type='histogram',
            name='Data',
            legendgroup='Data',
            x=_1d_array,
            marker=dict(color='#20d9ba'),
            histnorm='probability density',
            hoverinfo='x+y'))

    if plot_rug:

        data.append(
            dict(
                type='scatter',
                legendgroup='Data',
                showlegend=False,
                x=_1d_array,
                y=(0, ) * _1d_array.size,
                text=text,
                mode='markers',
                marker=dict(symbol='line-ns-open', color='#20d9ba'),
                hoverinfo='x+text'))

    grid = context_dict['grid']

    line_width = 3.2

    pdf = context_dict['pdf']

    data.append(
        dict(
            yaxis='y2',
            type='scatter',
            name='PDF',
            x=grid,
            y=pdf,
            line=dict(width=line_width, color='#24e7c0')))

    shape_pdf_reference = context_dict['shape_pdf_reference']

    shape_pdf_reference[pdf <= shape_pdf_reference] = None

    data.append(
        dict(
            yaxis='y2',
            type='scatter',
            name='Shape Reference',
            x=grid,
            y=shape_pdf_reference,
            line=dict(width=line_width, color='#9017e6')))

    location_pdf_reference = context_dict['location_pdf_reference']

    if location_pdf_reference is not None:

        location_pdf_reference[pdf <= location_pdf_reference] = None

        data.append(
            dict(
                yaxis='y2',
                type='scatter',
                name='Location Reference',
                x=grid,
                y=location_pdf_reference,
                line=dict(width=line_width, color='#4e40d8')))

    is_negative = context_dict['context_indices'] < 0

    for name, indices, color in (('- Context', is_negative, '#0088ff'),
                                 ('+ Context', ~is_negative, '#ff1968')):

        data.append(
            dict(
                yaxis='y2',
                type='scatter',
                name=name,
                x=grid[indices],
                y=absolute_context_indices[indices],
                line=dict(width=line_width, color=color),
                fill='tozeroy'))

    plot_and_save(
        dict(layout=layout, data=data), html_file_path, plotly_file_path)
