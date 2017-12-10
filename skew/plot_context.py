from os.path import join

from matplotlib.pyplot import close, figure, gcf, plot, show, ylim
from seaborn import distplot

from nd_array.nd_array.normalize_1d_array import normalize_1d_array

from .compute_context_indices import compute_context_indices
from .plot.plot.decorate import decorate
from .plot.plot.save_plot import save_plot


def plot_context(array_1d,
                 figure_size=(10, 10),
                 n_bin=80,
                 feature_name='Feature',
                 name='A Feature',
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
        array_1d (array): (n)
        figure_size (tuple):
        n_bin (int):
        feature_name (str): name of feature
        name (str): the name of this feature
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
    ylim(-1, 1)

    # ==========================================================================
    # Plot histogram
    # ==========================================================================
    distplot(
        array_1d,
        bins=n_bin,
        kde=False,
        norm_hist=True,
        hist_kws=dict(linewidth=0.92, color='#20D9BA', alpha=0.26))

    # ==========================================================================
    # Decorate
    # ==========================================================================
    decorate(style='white', title=name, xlabel=feature_name, ylabel='PDF')

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
        pdf_line_kwargs = dict(linestyle='-', linewidth=2.6)
        plot(
            grid,
            d['pdf'],
            # normalize_1d_array(d['pdf'], '0-1'),
            color='#20D9BA',
            **pdf_line_kwargs)
        plot(
            grid,
            d['pdf_reflection'],
            # normalize_1d_array(d['pdf_reflection'], '0-1'),
            color='#4E41D9',
            **pdf_line_kwargs)

    # ==========================================================================
    # Plot skew-t CDF
    # ==========================================================================
    if plot_skew_t_cdf:
        cdf_line_kwargs = dict(linestyle=':', linewidth=2.6)
        plot(grid, d['cdf'], color='#20D9BA', **cdf_line_kwargs)
        plot(grid, d['cdf_reflection'], color='#4E41D9', **cdf_line_kwargs)

    # ==========================================================================
    # Plot context indices
    # ==========================================================================
    if plot_context_indices:
        plot(
            grid,
            d['context_indices'],
            color='#FC154F',
            linestyle='-',
            linewidth=2.6)

    # ==========================================================================
    # Show and save
    # ==========================================================================
    if show_plot:
        show()
    if directory_path:
        save_plot(join(directory_path, 'plot_context', '{}.png'.format(name)))
    close()
