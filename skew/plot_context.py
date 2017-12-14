from os.path import join

from matplotlib.pyplot import close, figure, show

from .compute_context_indices import compute_context_indices
from .plot.plot.decorate import decorate
from .plot.plot.get_ax_positions_relative_to_ax import \
    get_ax_positions_relative_to_ax
from .plot.plot.plot_distribution import plot_distribution
from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE


def plot_context(array_1d,
                 title,
                 figure_size=FIGURE_SIZE,
                 n_bin=None,
                 plot_skew_t_pdf=True,
                 plot_context_indices=True,
                 n_grid=3000,
                 location=None,
                 scale=None,
                 df=None,
                 shape=None,
                 compute_context_indices_method='tail_reduction_reflection',
                 value_name='Value',
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
        n_grid (int):
        location (float):
        scale (float):
        df (float):
        shape (float):
        compute_context_indices_method (str): 'tail_reduction' | 'reflection' |
            'tail_reduction_reflection'
        value_name (str): the name of value
        show_plot (bool): whether to show plot
        directory_path (str): directory_path/context_plot/<title>.png will be
            saved
    Returns:
        None
    """

    # ==========================================================================
    # Set up figure
    # ==========================================================================
    ax = figure(figsize=figure_size).gca()

    # ==========================================================================
    # Plot histogram
    # ==========================================================================
    plot_distribution(
        array_1d,
        bins=n_bin,
        norm_hist=True,
        kde=False,
        hist_kws=dict(
            histtype='step',
            fill=True,
            linewidth=1.8,
            color='#003171',
            facecolor='#20D9BA',
            alpha=0.92,
            zorder=2),
        xlabel=value_name,
        ax=ax)

    context_dict = compute_context_indices(
        array_1d,
        n_grid=n_grid,
        location=location,
        scale=scale,
        df=df,
        shape=shape,
        compute_context_indices_method=compute_context_indices_method)

    grid = context_dict['grid']

    # ==========================================================================
    # Plot skew-t PDF and transformed PDF
    # ==========================================================================
    if plot_skew_t_pdf:

        pdf_backgdound_line_kwargs = dict(
            linestyle='-',
            linewidth=3.9,
            color='#003171',
            alpha=0.69,
            zorder=3)
        ax.plot(grid, context_dict['pdf'], **pdf_backgdound_line_kwargs)
        ax.plot(grid, context_dict['pdf_transformed'],
                **pdf_backgdound_line_kwargs)

        pdf_line_kwargs = dict(linestyle='-', linewidth=2.6, zorder=3)
        pdf = context_dict['pdf']
        ax.plot(grid, pdf, color='#20D9BA', **pdf_line_kwargs)
        ax.plot(
            grid,
            context_dict['pdf_transformed'],
            color='#9017E6',
            **pdf_line_kwargs)

    # ==========================================================================
    # Plot context indices
    # ==========================================================================
    if plot_context_indices:

        context_indices = context_dict['context_indices']
        is_negative = context_indices < 0

        context_indices_line_kwargs = dict(
            linestyle='-', linewidth=3.9, alpha=0.8, zorder=1)
        pdf_max = pdf.max()
        ax.fill_between(
            grid[is_negative],
            -1 * pdf_max * context_indices[is_negative],
            color='#0088FF',
            **context_indices_line_kwargs)
        ax.fill_between(
            grid[~is_negative],
            pdf_max * context_indices[~is_negative],
            color='#FF1968',
            **context_indices_line_kwargs)

    # ==========================================================================
    # Decorate
    # ==========================================================================
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

    decorate(ax=ax, style='white', title='Context Plot: {}'.format(title))

    # ==========================================================================
    # Show and save
    # ==========================================================================
    if directory_path:
        save_plot(join(directory_path, 'context_plot', '{}.png'.format(title)))
    if show_plot:
        show()
    close()
