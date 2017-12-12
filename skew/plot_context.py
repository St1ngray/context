from os.path import join

from matplotlib.pyplot import (close, figure, fill_between, gcf, plot, show,
                               xlim, ylim)
from seaborn import distplot

from .compute_context_indices import compute_context_indices
from .plot.plot.decorate import decorate
from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE


def plot_context(array_1d,
                 figure_size=FIGURE_SIZE,
                 n_bin=None,
                 plot_skew_t_pdf=True,
                 plot_skew_t_cdf=False,
                 plot_context_indices=True,
                 plot_both_context_on_top=True,
                 n_grid=3000,
                 location=None,
                 scale=None,
                 df=None,
                 shape=None,
                 title='Context Plot',
                 feature_name='Feature',
                 value_name='Value',
                 show_plot=True,
                 directory_path=None):
    """
    Plot context.
    Arguments:
        array_1d (array): (n)
        figure_size (tuple):
        n_bin (int):
        plot_skew_t_pdf (bool):
        plot_skew_t_cdf (bool):
        plot_context_indices (bool):
        plot_both_context_on_top (bool):
        n_grid (int):
        location (float):
        scale (float):
        df (float):
        shape (float):
        title (str):
        value_name (str): the name of value
        feature_name (str): the name of feature
        show_plot (bool): whether to show plot
        directory_path (str): directory_path//<id>.png will be saved
    Returns:
        None
    """

    # ==========================================================================
    # Set up figure
    # ==========================================================================
    figure(figsize=figure_size)
    xlim(array_1d.min(), array_1d.max())
    ylim([-1, 0][plot_both_context_on_top], 1)

    # ==========================================================================
    # Decorate
    # ==========================================================================
    decorate(
        style='white',
        title=title,
        xlabel=value_name,
        ylabel='Probability | Context Index')

    gcf().text(
        0.5,
        0.92,
        feature_name,
        size=18,
        weight='bold',
        color='#20D9BA',
        horizontalalignment='center')

    # ==========================================================================
    # Plot histogram
    # ==========================================================================
    if not n_bin:
        n_bin = array_1d.size // 8

    distplot(
        array_1d,
        bins=n_bin,
        kde=False,
        norm_hist=True,
        hist_kws=dict(
            histtype='step',
            fill=True,
            linewidth=0.92,
            color='#003171',
            facecolor='#20D9BA',
            alpha=0.92,
            zorder=2))

    if plot_skew_t_pdf or plot_skew_t_cdf or plot_context_indices:
        d = compute_context_indices(
            array_1d,
            n_grid=n_grid,
            location=location,
            scale=scale,
            df=df,
            shape=shape)

        gcf().text(
            0.5,
            0.9,
            'N={:.0f}    Location={:.2f}    Scale={:.2f}    DF={:.2f}    Shape={:.2f}'.
            format(*d['fit']),
            size=16,
            weight='bold',
            color='#220530',
            horizontalalignment='center')

        grid = d['grid']

    # ==========================================================================
    # Plot skew-t PDF
    # ==========================================================================
    if plot_skew_t_pdf:

        pdf_backgdound_line_kwargs = dict(
            linestyle='-',
            linewidth=6.9,
            color='#003171',
            alpha=0.69,
            zorder=3)
        plot(grid, d['pdf'], **pdf_backgdound_line_kwargs)
        plot(grid, d['pdf_reflection'], **pdf_backgdound_line_kwargs)

        pdf_line_kwargs = dict(linestyle='-', linewidth=3.9, zorder=3)
        plot(grid, d['pdf'], color='#20D9BA', **pdf_line_kwargs)
        plot(grid, d['pdf_reflection'], color='#9017E6', **pdf_line_kwargs)

    # ==========================================================================
    # Plot skew-t CDF
    # ==========================================================================
    if plot_skew_t_cdf:

        cdf_line_kwargs = dict(linestyle=':', linewidth=3.9, zorder=4)
        plot(grid, d['cdf'], color='#20D9BA', **cdf_line_kwargs)
        plot(grid, d['cdf_reflection'], color='#9017E6', **cdf_line_kwargs)

    # ==========================================================================
    # Plot context indices
    # ==========================================================================
    if plot_context_indices:

        context_indices = d['context_indices']
        is_negative = context_indices < 0

        context_indices_line_kwargs = dict(
            linestyle='-', linewidth=3.9, alpha=0.8, zorder=1)
        fill_between(
            grid[is_negative],
            [1, -1][plot_both_context_on_top] * context_indices[is_negative],
            color='#0088FF',
            **context_indices_line_kwargs)

        fill_between(
            grid[~is_negative],
            context_indices[~is_negative],
            color='#FF1968',
            **context_indices_line_kwargs)

    # ==========================================================================
    # Show and save
    # ==========================================================================
    if directory_path:
        save_plot(
            join(directory_path, 'context_plot', '{}.png'.format(
                feature_name)))
    if show_plot:
        show()
    close()
