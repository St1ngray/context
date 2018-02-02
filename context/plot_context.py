from os.path import join

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import figure, subplot
from numpy import absolute
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
                 n_grid=1000,
                 degree_of_freedom_for_tail_reduction=10e8,
                 global_location=None,
                 global_scale=None,
                 add_context_summary_to_title=True,
                 xlabel='Value',
                 directory_path=None):
    """
    Plot context.
    Arguments:
        array_1d (ndarray): (n, )
        title (str):
        figure_size (iterable):
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
        add_context_summary_to_title (bool):
        xlabel (str):
        directory_path (str):
    Returns:
    """

    figure(figsize=figure_size)

    gridspec = GridSpec(100, 1)

    i = 82
    ax = subplot(gridspec[:i, :])
    ax_bottom = subplot(gridspec[i:, :], sharex=ax)

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

    grid = context_dict['grid']
    pdf = context_dict['pdf']
    context_indices = context_dict['context_indices']

    ax.set_ylim(-0.1,
                1.008 * max(1, pdf.max(), absolute(context_indices).max()))

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
        positive_context_indices_color = '#FF1968'
        negative_context_indices_color = '#0088FF'

        r_context_indices_alpha = 0.69
        s_context_indices_alpha = 0.22

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

        if add_context_summary_to_title:
            title += ' (Context Summary {:.2f})'.format(
                context_dict['context_summary'])

    ax_x_min, ax_x_max, ax_y_min, ax_y_max = get_ax_positions(ax, 'ax')

    ax.text(
        (ax_x_min + ax_x_max) / 2,
        ax_y_max * 1.08,
        title,
        horizontalalignment='center',
        **FONT_LARGEST)

    if plot_fit_and_references:
        ax.text(
            (ax_x_min + ax_x_max) / 2,
            ax_y_max * 1.022,
            'N={:.0f}   Location={:.2f}   Scale={:.2f}   Degree of Freedom={:.2f}   Shape={:.2f}'.
            format(*context_dict['fit']),
            horizontalalignment='center',
            **FONT_STANDARD)

    decorate_ax(
        ax, despine_kwargs={
            'bottom': True,
        }, style='white')

    swarmplot_kwargs = {
        's': 5.1,
    }
    swarmplot(x=array_1d, ax=ax_bottom, color=data_color, **swarmplot_kwargs)

    decorate_ax(
        ax_bottom, despine_kwargs={
            'left': True,
        }, xlabel=xlabel, yticks=())

    if directory_path:
        save_plot(
            join(directory_path, 'context_plot',
                 clean_name('{}.png'.format(title))))
