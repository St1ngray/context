from os.path import join

from numpy import absolute, histogram, isnan, nanmean

from .compute_context import compute_context
from .nd_array.nd_array.compute_n_histogram_bin_using_freedman_diaconis import \
    compute_n_histogram_bin_using_freedman_diaconis
from .support.support.path import clean_name


def plot_context(array_1d,
                 title,
                 plot_fit_and_references=True,
                 plot_context_indices=True,
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
                 y_max_is_histogram_max=True,
                 plot_swarm=True,
                 xlabel='Value',
                 directory_path=None):
    """
    Plot context.
    Arguments:
        array_1d (ndarray): (n, )
        title (str):
        plot_fit_and_references (bool):
        plot_context_indices (bool):
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
        y_max_is_histogram_max (bool):
        plot_swarm (bool):
        xlabel (str):
        directory_path (str):
    Returns:
    """

    array_1d = array_1d.copy()
    is_nan = isnan(array_1d)
    if is_nan.all():
        raise ValueError('array_1d has only nan.')
    else:
        array_1d[is_nan] = nanmean(array_1d)

    n_bin = compute_n_histogram_bin_using_freedman_diaconis(array_1d)

    if plot_swarm:
        i = 0.8
    else:
        i = 1

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

    histogram_max = histogram(array_1d, bins=n_bin, normed=True)[0].max()
    pdf_max = context_dict['pdf'].max()
    context_indices_absolute_max = absolute(
        context_dict['context_indices']).max()
    if y_max_is_histogram_max:
        y_max = max(histogram_max, pdf_max)
    else:
        y_max = max(histogram_max, pdf_max, context_indices_absolute_max)

    data_color = '#20D9BA'
    plot_distribution(
        array_1d,
        ax=ax,
        distplot_kwargs={
            'bins': n_bin,
            'norm_hist': True,
            'hist_kws': {
                'zorder': 1,
                'histtype': 'step',
                'fill': True,
                'linewidth': 1.8,
                'color': '#23191E',
                'facecolor': data_color,
                'alpha': 1,
            },
        })

    linewidth = 3.2
    background_linewidth = 6.4
    background_color = '#EBF6F7'

    if plot_fit_and_references:

        z_order = 5
        ax.plot(
            context_dict['grid'],
            context_dict['pdf'],
            zorder=z_order,
            linewidth=background_linewidth,
            color=background_color)
        ax.plot(
            context_dict['grid'],
            context_dict['pdf'],
            zorder=z_order,
            linewidth=linewidth,
            color=data_color)

        z_order = 3
        ax.plot(
            context_dict['grid'],
            context_dict['r_pdf_reference'],
            zorder=z_order,
            linewidth=background_linewidth,
            color=background_color)
        ax.plot(
            context_dict['grid'],
            context_dict['r_pdf_reference'],
            zorder=z_order,
            linewidth=linewidth,
            color='#9017E6')

        if context_dict['s_pdf_reference'] is not None:

            z_order = 4
            ax.plot(
                context_dict['grid'],
                context_dict['s_pdf_reference'],
                zorder=z_order,
                linewidth=background_linewidth,
                color=background_color)
            ax.plot(
                context_dict['grid'],
                context_dict['s_pdf_reference'],
                zorder=z_order,
                linewidth=linewidth,
                color='#4E40D8')

    if plot_context_indices:

        y = absolute(context_dict['context_indices'])
        if y_max_is_histogram_max and y_max < context_indices_absolute_max:
            y /= context_indices_absolute_max
            y *= y_max

        z_order = 2
        ax.plot(
            context_dict['grid'],
            y,
            zorder=z_order,
            linewidth=background_linewidth,
            color=background_color)
        ax.plot(
            context_dict['grid'],
            y,
            zorder=z_order,
            linewidth=linewidth,
            color='#4C221B')

        r_context_indices_alpha = 0.69
        s_context_indices_alpha = 0.22

        positive_context_indices = 0 <= context_dict['context_indices']

        y0 = context_dict['context_indices'][positive_context_indices]
        y1 = context_dict['r_context_indices'][positive_context_indices]
        if y_max_is_histogram_max and y_max < context_indices_absolute_max:
            if y0.size:
                y0 /= context_indices_absolute_max
                y0 *= y_max
            if y1.size:
                y1 /= context_indices_absolute_max
                y1 *= y_max

        positive_context_indices_color = '#FF1968'
        ax.fill_between(
            context_dict['grid'][positive_context_indices],
            y0,
            y1,
            zorder=z_order,
            linewidth=linewidth,
            color=positive_context_indices_color,
            alpha=s_context_indices_alpha)
        ax.fill_between(
            context_dict['grid'][positive_context_indices],
            y1,
            zorder=z_order,
            linewidth=linewidth,
            color=positive_context_indices_color,
            alpha=r_context_indices_alpha)

        y0 = -context_dict['context_indices'][~positive_context_indices]
        y1 = absolute(
            context_dict['r_context_indices'])[~positive_context_indices]
        if y_max_is_histogram_max and y_max < context_indices_absolute_max:
            if y0.size:
                y0 /= context_indices_absolute_max
                y0 *= y_max
            if y1.size:
                y1 /= context_indices_absolute_max
                y1 *= y_max

        negative_context_indices_color = '#0088FF'
        ax.fill_between(
            context_dict['grid'][~positive_context_indices],
            y0,
            y1,
            zorder=z_order,
            linewidth=linewidth,
            color=negative_context_indices_color,
            alpha=s_context_indices_alpha)
        ax.fill_between(
            context_dict['grid'][~positive_context_indices],
            y1,
            zorder=z_order,
            linewidth=linewidth,
            color=negative_context_indices_color,
            alpha=r_context_indices_alpha)

    ax_x_min, ax_x_max, ax_y_min, ax_y_max = get_ax_positions(ax, 'ax')

    ax.text(
        (ax_x_min + ax_x_max) / 2,
        ax_y_max * 1.16,
        title,
        horizontalalignment='center',
        **FONT_LARGEST)

    if plot_fit_and_references:
        ax.text(
            (ax_x_min + ax_x_max) / 2,
            ax_y_max * 1.08,
            'N={:.0f}   Location={:.2f}   Scale={:.2f}   Degree of Freedom={:.2f}   Shape={:.2f}'.
            format(*context_dict['fit']),
            horizontalalignment='center',
            **FONT_STANDARD)

    if plot_context_indices:
        ax.text(
            (ax_x_min + ax_x_max) / 2,
            ax_y_max * 1.02,
            'Negative Context Summary={:.2f}   Positive Context Summary={:.2f}'.
            format(context_dict['negative_context_summary'],
                   context_dict['positive_context_summary']),
            horizontalalignment='center',
            **FONT_STANDARD)

    decorate_ax(
        ax, despine_kwargs={
            'bottom': plot_swarm,
        }, xlabel=xlabel)

    if plot_swarm:
        s = 5.6
        swarmplot(x=array_1d, ax=ax_bottom, s=s, color=data_color)
        decorate_ax(
            ax_bottom,
            despine_kwargs={
                'left': True,
            },
            xlabel=xlabel,
            yticks=())

    if directory_path:
        save_plot(
            join(directory_path, 'context_plot',
                 clean_name('{}.png'.format(title))))
