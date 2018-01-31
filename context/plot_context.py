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
                 n_grid=3000,
                 degree_of_freedom_for_tail_reduction=10e8,
                 global_shape=None,
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
        global_shape (float):
        global_location (float):
        global_scale (float):
        add_context_summary_to_title (bool):
        xlabel (str):
        directory_path (str):
    Returns:
    """

    figure(figsize=figure_size)

    gridspec = GridSpec(100, 1)

    ax = subplot(gridspec[:80, :])
    ax_bottom = subplot(gridspec[80:, :], sharex=ax)

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
        global_shape=global_shape,
        global_location=global_location,
        global_scale=global_scale)

    fit = context_dict['fit']
    grid = context_dict['grid']
    pdf = context_dict['pdf']
    pdf_reference = context_dict['pdf_reference']
    global_pdf_reference = context_dict['global_pdf_reference']
    context_indices = context_dict['context_indices']
    context_summary = context_dict['context_summary']

    if global_pdf_reference is None:
        global_pdf_reference_max = 0
    else:
        global_pdf_reference_max = global_pdf_reference.max()

    ax.set_ylim(0,
                max((
                    pdf.max(),
                    pdf_reference.max(),
                    global_pdf_reference_max,
                    absolute(context_indices).max(),
                    1, )) * 1.08)

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
                'facecolor': '#20D9BA',
                'alpha': 0.8,
                'zorder': 2,
            },
        })

    linewidth = 5.1

    if plot_fit_and_references:

        pdf_backgdound_line_kwargs = {
            'linestyle': '-',
            'linewidth': linewidth * 1.51,
            'color': '#EBF6F7',
            'alpha': 0.8,
            'zorder': 3,
        }
        pdf_line_kwargs = {
            'linestyle': '-',
            'linewidth': linewidth,
            'zorder': 4,
        }

        ax.plot(grid, pdf, **pdf_backgdound_line_kwargs)
        ax.plot(grid, pdf, color='#20D9BA', **pdf_line_kwargs)

        ax.plot(grid, pdf_reference, **pdf_backgdound_line_kwargs)
        ax.plot(grid, pdf_reference, color='#9017E6', **pdf_line_kwargs)

        if global_pdf_reference is not None:

            ax.plot(grid, global_pdf_reference, **pdf_backgdound_line_kwargs)
            ax.plot(
                grid, global_pdf_reference, color='#4E40D8', **pdf_line_kwargs)

    if plot_context_indices:

        is_positive = 0 <= context_indices

        context_indices_line_kwargs = {
            'linestyle': '-',
            'linewidth': linewidth,
            'alpha': 0.8,
            'zorder': 1,
        }

        ax.fill_between(
            grid[is_positive],
            context_indices[is_positive],
            color='#FF1968',
            **context_indices_line_kwargs)

        ax.fill_between(
            grid[~is_positive],
            -1 * context_indices[~is_positive],
            color='#0088FF',
            **context_indices_line_kwargs)

        if add_context_summary_to_title:
            title += ' (Context Summary={:.2f})'.format(context_summary)

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
            format(*fit),
            horizontalalignment='center',
            **FONT_STANDARD)

    decorate_ax(
        ax, despine_kwargs={
            'bottom': True,
        }, style='white')

    swarmplot(x=array_1d, ax=ax_bottom, color='#20D9BA', alpha=0.8)
    # TODO: enable multiple swarmplots (make sure to dynamically set axes)
    # for annotation_i, annotation_vector in annotation_x_sample.iterrows():
    #     swarmplot(
    #         x=annotation_vector * array_1d,
    #         ax=ax_bottom,
    #         color=color,
    #         alpha=0.8)
    decorate_ax(
        ax_bottom, despine_kwargs={
            'left': True,
        }, xlabel=xlabel, yticks=())

    if directory_path:
        save_plot(
            join(directory_path, 'context_plot',
                 clean_name('{}.png'.format(title))))
