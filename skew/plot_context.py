from os.path import join

from matplotlib.pyplot import close, figure, show

from .compute_context import compute_context
from .plot.plot.decorate import decorate
from .plot.plot.get_ax_positions_relative_to_ax import \
    get_ax_positions_relative_to_ax
from .plot.plot.plot_distribution import plot_distribution
from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE
from .support.support.path import clean_file_name


def plot_context(array_1d,
                 title,
                 figure_size=[FIGURE_SIZE[0] * 1.88, FIGURE_SIZE[1]],
                 n_bin=None,
                 plot_skew_t_pdf=True,
                 plot_context_indices=True,
                 location=None,
                 scale=None,
                 df=None,
                 shape=None,
                 n_grid=3000,
                 compute_context_method='tail_reduction_reflection',
                 degrees_of_freedom_for_tail_reduction=10e8,
                 summarize_context_by='absolute_value_weighted_context',
                 summarize_context_side='shape_side',
                 xlabel='Value',
                 show_plot=True,
                 directory_path=None):
    """
    Plot context.
    Arguments:
        array_1d (array): (n)
        title (str):
        figure_size (iterable):
        n_bin (int):
        plot_skew_t_pdf (bool):
        plot_context_indices (bool):
        location (float):
        scale (float):
        df (float):
        shape (float):
        n_grid (int):
        compute_context_method (str): 'tail_reduction_reflection' |
            'tail_reduction' | 'reflection' |
        degrees_of_freedom_for_tail_reduction (number):
        summarize_context_by (str): 'absolute_value_weighted_context' |
            'context'
        summarize_context_side (str): 'shape_side' | 'both_sides'
        xlabel (str):
        show_plot (bool):
        directory_path (str):
    Returns:
        None
    """

    ax = figure(figsize=figure_size).gca()

    plot_distribution(
        array_1d,
        bins=n_bin,
        kde=False,
        hist_kws=dict(
            histtype='step',
            fill=True,
            linewidth=1.8,
            color='#003171',
            facecolor='#20D9BA',
            alpha=0.92,
            zorder=2),
        norm_hist=True,
        ax=ax,
        xlabel=xlabel)

    context_dict = compute_context(
        array_1d,
        location=location,
        scale=scale,
        df=df,
        shape=shape,
        n_grid=n_grid,
        compute_context_method=compute_context_method,
        degrees_of_freedom_for_tail_reduction=
        degrees_of_freedom_for_tail_reduction,
        summarize_context_by=summarize_context_by,
        summarize_context_side=summarize_context_side)

    grid = context_dict['grid']
    pdf = context_dict['pdf']

    linewidth = 3.9

    if plot_skew_t_pdf:
        plot_skew_t_pdf_zorder = 3
        background_line_alpha = 0.69
        background_line_color = '#003171'

        pdf_backgdound_line_kwargs = dict(
            linestyle='-',
            linewidth=linewidth * 1.8,
            color=background_line_color,
            alpha=background_line_alpha,
            zorder=plot_skew_t_pdf_zorder)
        pdf_line_kwargs = dict(
            linestyle='-', linewidth=linewidth, zorder=plot_skew_t_pdf_zorder)

        ax.plot(grid, pdf, **pdf_backgdound_line_kwargs)
        ax.plot(grid, pdf, color='#20D9BA', **pdf_line_kwargs)

        ax.plot(grid, context_dict['pdf_transformed'],
                **pdf_backgdound_line_kwargs)
        ax.plot(
            grid,
            context_dict['pdf_transformed'],
            color='#9017E6',
            **pdf_line_kwargs)

    if plot_context_indices:
        context_indices = context_dict['context_indices']
        is_positive = 0 < context_indices

        context_indices_line_kwargs = dict(
            linestyle='-', linewidth=linewidth, alpha=0.8, zorder=1)

        pdf_max = pdf.max()

        ax.fill_between(
            grid[is_positive],
            pdf_max * context_indices[is_positive],
            color='#FF1968',
            **context_indices_line_kwargs)

        ax.fill_between(
            grid[~is_positive],
            context_indices[~is_positive] * pdf_max * -1,
            color='#0088FF',
            **context_indices_line_kwargs)

        title = '{} (Context Summary = {:.2f})'.format(
            title, context_dict['context_summary'])

    if plot_skew_t_pdf or plot_context_indices:
        ax_x_min, ax_x_max, ax_y_min, ax_y_max = get_ax_positions_relative_to_ax(
            ax)

        ax.text(
            (ax_x_min + ax_x_max) / 2,
            ax_y_max * 1.026,
            'N={:.0f}  Location={:.2f}  Scale={:.2f}  DF={:.2f}  Shape={:.2f}'.
            format(*context_dict['fit']),
            size=16,
            weight='bold',
            color='#181B26',
            horizontalalignment='center')

    decorate(ax=ax, style='white', title=title)

    if directory_path:
        save_plot(
            join(directory_path, 'context_plot',
                 clean_file_name('{}.png'.format(title))))
    if show_plot:
        show()
    close()
