from os.path import join

from matplotlib.pyplot import close, figure, gcf, plot, show
from seaborn import distplot

from .compute_context_indices import compute_context_indices
from .plot.plot.decorate import decorate
from .plot.plot.save_plot import save_plot
from .plot.plot.style import FIGURE_SIZE


def plot_context(series,
                 figure_size=FIGURE_SIZE,
                 n_bin=80,
                 feature_name='Feature',
                 plot_skew_t_pdf=True,
                 plot_skew_t_cdf=True,
                 plot_context_indices=True,
                 n_grid=3000,
                 location=None,
                 scale=None,
                 df=None,
                 shape=None,
                 show_plot=True,
                 directory_path=None):
    """
    Plot context.
    Arguments:
        series (DataFrame): (n_sample)
        figure_size (tuple):
        n_bin (int):
        feature_name (str):
        plot_skew_t_pdf (bool):
        plot_skew_t_cdf (bool):
        plot_context_indices (bool):
        n_grid (int):
        location (float):
        scale (float):
        df (float):
        shape (float):
        show_plot (bool): whether to show plot
        directory_path (str): directory_path//<id>.png will be saved
    Returns:
        None
    """

    # ==========================================================================
    # Set up figure
    # ==========================================================================
    figure(figsize=figure_size)

    # ==========================================================================
    # Plot histogram
    # ==========================================================================
    distplot(
        series,
        bins=n_bin,
        kde=False,
        norm_hist=True,
        hist_kws=dict(linewidth=0.92, color='#20D9BA', alpha=0.26))

    # ==========================================================================
    # Decorate
    # ==========================================================================
    decorate(
        style='white', title=series.name, xlabel=feature_name, ylabel='PDF')

    if plot_skew_t_pdf or plot_skew_t_cdf or plot_context_indices:
        grids, pdf, pdf_reflection, cdf, cdf_reflection, context_indices = compute_context_indices(
            series,
            n_grid=n_grid,
            location=location,
            scale=scale,
            df=df,
            shape=shape)
        gcf().text(
            0.5,
            0.9,
            'N={:.0f}    Location={:.2f}    Scale={:.2f}    DF={:.2f}    Shape={:.2f}'.
            format(series.size, location, scale, df, shape),
            size=16,
            weight='bold',
            color='#220530',
            horizontalalignment='center')
        line_kwargs = dict(linestyle='-', linewidth=2.6)

    # ==========================================================================
    # Plot skew-t PDF
    # ==========================================================================
    if plot_skew_t_pdf:
        plot(grids, pdf, color='#20D9BA', **line_kwargs)
        plot(grids, pdf_reflection, color='#4E41D9', **line_kwargs)

    # ==========================================================================
    # Plot skew-t CDF
    # ==========================================================================
    if plot_skew_t_cdf:
        plot(grids, cdf, linestyle='.', color='#20D9BA', **line_kwargs)
        plot(
            grids,
            cdf_reflection,
            linestyle='.',
            color='#4E41D9',
            **line_kwargs)

    # ==========================================================================
    # Plot context index
    # ==========================================================================
    if plot_context_indices:
        plot(grids, context_indices, color='#FC154F', **line_kwargs)

    # ==========================================================================
    # Show and save
    # ==========================================================================
    if show_plot:
        show()
    if directory_path:
        save_plot(
            join(directory_path, 'plot_context', '{}.png'.format(series.name)))
    close()
