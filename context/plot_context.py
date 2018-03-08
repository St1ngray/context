from os.path import join

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, gca, subplot
from numpy import absolute, isnan, nanmean
from seaborn import swarmplot

from .compute_context import compute_context
from .plot.plot.decorate_ax import decorate_ax
from .plot.plot.get_ax_positions import get_ax_positions
from .plot.plot.plot_distribution import plot_distribution
from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE, FONT_LARGEST, FONT_STANDARD
from .support.support.path import clean_name


def plot_context(array_1d,
                 title,
                 figure_size=(
                     FIGURE_SIZE[0] * 1.80,
                     FIGURE_SIZE[1], ),
                 y_max_is_1=False,
                 n_bin=None,
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
                 degree_of_freedom_for_tail_reduction=10e8,
                 global_location=None,
                 global_scale=None,
                 xlabel='Value',
                 directory_path=None):
    """
    Plot context.
    Arguments:
        array_1d (ndarray): (n, )
        title (str):
        figure_size (iterable):
        y_max_is_1 (bool):
        n_bin (int):
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

    figure(figsize=figure_size)

    gridspec = GridSpec(100, 1)

    if array_1d.size < 10000:
        i = 82
        ax = subplot(gridspec[:i, :])
        ax_bottom = subplot(gridspec[i:, :], sharex=ax)
    else:
        ax = gca()

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

    pdf = context_dict['pdf']
    context_indices = context_dict['context_indices']

    if y_max_is_1:
        y_max = 1
    else:
        y_max = max(1, pdf.max(), absolute(context_indices).max())
    ax.set_ylim(-0.1, 1.008 * y_max)

    data_color = '#20D9BA'
    plot_distribution(
        array_1d,
        ax=ax,
        distplot_kwargs={
            'bins': n_bin,
            'norm_hist': True,
            'hist_kws': {
                'histtype': 'step',
                'fill': True,
                'linewidth': 1.8,
                'color': '#23191E',
                'facecolor': data_color,
                'alpha': 1,
                'zorder': 1,
            },
        })

    grid = context_dict['grid']
    linewidth = 5.1

    background_line_kwargs = {
        'linewidth': linewidth * 1.51,
        'color': '#EBF6F7',
    }
    line_kwargs = {
        'linewidth': linewidth,
    }
    if plot_fit_and_references:

        z_order = 5
        ax.plot(grid, pdf, zorder=z_order, **background_line_kwargs)
        ax.plot(grid, pdf, color=data_color, zorder=z_order, **line_kwargs)

        r_pdf_reference = context_dict['r_pdf_reference']
        r_color = '#9017E6'
        z_order = 3
        ax.plot(
            grid, r_pdf_reference, zorder=z_order, **background_line_kwargs)
        ax.plot(
            grid,
            r_pdf_reference,
            color=r_color,
            zorder=z_order,
            **line_kwargs)

        s_pdf_reference = context_dict['s_pdf_reference']
        if s_pdf_reference is not None:

            s_color = '#4E40D8'
            z_order = 4
            ax.plot(
                grid,
                s_pdf_reference,
                zorder=z_order,
                **background_line_kwargs)
            ax.plot(
                grid,
                s_pdf_reference,
                color=s_color,
                zorder=z_order,
                **line_kwargs)

    if plot_context_indices:

        z_order = 2
        ax.plot(
            grid,
            absolute(context_indices),
            zorder=z_order,
            **background_line_kwargs)
        ax.plot(
            grid,
            absolute(context_indices),
            color='#4C221B',
            zorder=z_order,
            **line_kwargs)

        r_context_indices = context_dict['r_context_indices']

        context_indices_line_kwargs = {
            'linewidth': linewidth,
            'zorder': z_order,
        }

        positive_context_indices = 0 <= context_indices

        r_context_indices_alpha = 0.69
        s_context_indices_alpha = 0.22

        positive_context_indices_color = '#FF1968'
        ax.fill_between(
            grid[positive_context_indices],
            context_indices[positive_context_indices],
            r_context_indices[positive_context_indices],
            color=positive_context_indices_color,
            alpha=s_context_indices_alpha,
            **context_indices_line_kwargs)
        ax.fill_between(
            grid[positive_context_indices],
            r_context_indices[positive_context_indices],
            color=positive_context_indices_color,
            alpha=r_context_indices_alpha,
            **context_indices_line_kwargs)

        negative_context_indices_color = '#0088FF'
        ax.fill_between(
            grid[~positive_context_indices],
            -context_indices[~positive_context_indices],
            absolute(r_context_indices)[~positive_context_indices],
            color=negative_context_indices_color,
            alpha=s_context_indices_alpha,
            **context_indices_line_kwargs)
        ax.fill_between(
            grid[~positive_context_indices],
            absolute(r_context_indices)[~positive_context_indices],
            color=negative_context_indices_color,
            alpha=r_context_indices_alpha,
            **context_indices_line_kwargs)

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
            'bottom': True,
        }, style='white')

    if array_1d.size < 10000:
        swarmplot_kwargs = {
            's': 5.1,
        }
        swarmplot(
            x=array_1d, ax=ax_bottom, color=data_color, **swarmplot_kwargs)

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
