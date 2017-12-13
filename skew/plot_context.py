from os.path import join

from matplotlib.pyplot import (close, figure, fill_between, gcf, hist, plot,
                               show, xlim, ylim)
from numpy import ones_like

from .compute_context_indices import compute_context_indices
from .plot.plot.decorate import decorate
from .plot.plot.save_plot import save_plot


def plot_context(array_1d,
                 figure_size=(8, 8),
                 n_bin=None,
                 plot_skew_t_pdf=True,
                 plot_context_indices=True,
                 plot_both_context_on_top=True,
                 n_grid=3000,
                 location=None,
                 scale=None,
                 df=None,
                 shape=None,
                 compute_context_indices_method='tail_reduction_reflection',
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
        plot_context_indices (bool):
        plot_both_context_on_top (bool):
        n_grid (int):
        location (float):
        scale (float):
        df (float):
        shape (float):
        compute_context_indices_method (str): 'tail_reduction' | 'reflection' |
            'tail_reduction_reflection'
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
    hist(
        array_1d,
        weights=ones_like(array_1d) / array_1d.size,
        bins=n_bin,
        histtype='step',
        fill=True,
        linewidth=0.92,
        color='#003171',
        facecolor='#20D9BA',
        alpha=0.92,
        zorder=2)

    if plot_skew_t_pdf or plot_context_indices:
        d = compute_context_indices(
            array_1d,
            n_grid=n_grid,
            location=location,
            scale=scale,
            df=df,
            shape=shape,
            compute_context_indices_method=compute_context_indices_method)

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
    # Plot skew-t PDF and transformed PDF
    # ==========================================================================
    if plot_skew_t_pdf:

        pdf_backgdound_line_kwargs = dict(
            linestyle='-',
            linewidth=3.9,
            color='#003171',
            alpha=0.69,
            zorder=3)
        plot(grid, d['pdf'], **pdf_backgdound_line_kwargs)
        plot(grid, d['pdf_transformed'], **pdf_backgdound_line_kwargs)

        pdf_line_kwargs = dict(linestyle='-', linewidth=2.6, zorder=3)
        plot(grid, d['pdf'], color='#20D9BA', **pdf_line_kwargs)
        plot(grid, d['pdf_transformed'], color='#9017E6', **pdf_line_kwargs)

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
