from os.path import join

from matplotlib.pyplot import close, figure, gca, plot, show
from numpy import argmin, asarray, cumsum, empty, linspace, log, sign, where
from pandas import DataFrame
from seaborn import distplot
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .nd_array.nd_array.get_coordinates_for_reflection import \
    get_coordinates_for_reflection
from .nd_array.nd_array.normalize_1d_array import normalize_1d_array
from .plot.plot.decorate import decorate
from .plot.plot.save_plot import save_plot
from .plot.plot.style import CMAP_CATEGORICAL_TAB20, FIGURE_SIZE


def plot_essentiality(feature_x_sample,
                      feature_x_fit,
                      enumerate_functions=False,
                      figure_size=FIGURE_SIZE,
                      n_x_grid=3000,
                      n_bins=80,
                      feature_name='Feature',
                      plot_fits=True,
                      show_plot=True,
                      directory_path=None):
    """
    Make essentiality plot for each gene.
    Arguments:
        feature_x_sample (DataFrame): (n_feature, n_sample)
        feature_x_fit (DataFrame): (n_feature, 5 [N, DF, Shape, Location,
            Scale])
        enumerate_functions (bool): whether to plot fit functions
        figure_size (tuple): figure size
        n_x_grid (int): number of x grids
        n_bins (int): number of histogram bins
        feature_name (str): feature name
        plot_fits (bool): whether to plot fitted lines
        show_plot (bool): whether to show plot
        directory_path (str): directory_path//<id>.png will be saved
    Returns:
        None
    """

    # ==========================================================================
    # Plot each feature
    # ==========================================================================
    for i, (f_i, f_v) in enumerate(feature_x_sample.iterrows()):
        print('({}/{}) {} ...'.format(i, feature_x_sample.shape[0], f_i))

        # ======================================================================
        # Set up figure
        # ======================================================================
        # Initialize a figure
        fig = figure(figsize=figure_size)
        ax_graph = gca()

        # ======================================================================
        # Plot histogram
        # ======================================================================
        distplot(
            f_v,
            bins=n_bins,
            kde=False,
            norm_hist=True,
            hist_kws=dict(linewidth=0.92, color='#20D9BA', alpha=0.26),
            ax=ax_graph)

        # ==============================================================
        # Decorate
        # ==============================================================
        decorate(
            ax=ax_graph,
            style='white',
            title=f_i,
            xlabel=feature_name,
            ylabel='Frequency')

        # ==================================================================
        # Plot skew-t fit PDF
        # ==================================================================
        # Initialize a skew-t generator
        skew_t = ACSkewT_gen()

        # Set up grids
        grids = linspace(f_v.min(), f_v.max(), n_x_grid)

        # Parse fitted parameters
        n, df, shape, location, scale = feature_x_fit[
            f_i, ['N', 'Location', 'Scale', 'DF', 'Shape']]
        fig.text(
            0.5,
            0.9,
            'N={:.0f}    Location={:.2f}    Scale={:.2f}    DF={:.2f}    Shape={:.2f}'.
            format(n, location, scale, df, shape),
            size=16,
            weight='bold',
            color='#220530',
            horizontalalignment='center')

        # Generate skew-t PDF
        skew_t_pdf = skew_t.pdf(grids, df, shape, loc=location, scale=scale)

        # Plot skew-t PDF
        line_kwargs = dict(linestyle='-', linewidth=2.6)
        ax_graph.plot(grids, skew_t_pdf, color='#20D9BA', **line_kwargs)

        # ==================================================================
        # Plot reflected skew-t PDF
        # ==================================================================
        # Generate skew-t PDF over reflected grids
        skew_t_pdf_r = skew_t.pdf(
            get_coordinates_for_reflection(skew_t_pdf, grids),
            df,
            shape,
            loc=location,
            scale=scale)

        # Plot over the original grids
        ax_graph.plot(grids, skew_t_pdf_r, color='#4E41D9', **line_kwargs)

        # ==================================================================
        # Plot essentiality indices from various functions
        # ==================================================================
        figure_size_ = (asarray(figure_size) * 0.7).astype(int)
        if enumerate_functions:
            functions = [
                # f1 /f2
                # Explode 'f1 / f2',
                # Signal at center 'log(f1 / f2)',
                # Explode 'where(f2 < f1, f1 / f2, 0)',
                # Not that good during entropy test 'where(f2 < f1, log(f1 /
                # f2), 0)',

                # - f2 /f1
                # Signal at center '-(f2 / f1)',
                # Signal at center '-log(f2 / f1)',
                # Spikes to 0 after center 'where(f2 < f1, -(f2 / f1), 0)',
                # == log(f1/ f2) 'where(f2 < f1, -log(f2 / f1), 0)',

                # carea1 / carea2
                # Explode 'carea1 / carea2',
                # Not that good during entropy test 'log(carea1 / carea2)',
                # Explode 'where(f2 < f1, carea1 / carea2, 0)',
                # 0ing abruptly drops 'where(f2 < f1, log(carea1 / carea2), 0)'

                # (f1 - f2) / f1
                # Better during only f2 < f1 '(f1 - f2) / f1',
                # Normalized same as not logging and raising to a power'log(
                # (f1 - f2) / f1 )',
                'where(f2 < f1, (f1 - f2) / f1, 0)',
                # Spikes to 0 after center 'where(f2 < f1, log( (f1 - f2) /
                # f1 ), 0)',

                # ((f1 - f2) / f1)^scale
                # Super negative '((f1 - f2) / f1)**{}'.format(scale),
                'where(f2 < f1log, ((f1 - f2) / f1)**{}, 0)'.format(scale),
                # log
                # Same as just log 'where(f2 < f1, log( ((f1 - f2) / f1)**{}
                # ), 0 )'.format(scale),

                # Hard to interpret # ((f1 - f2) / f1)^(1/scale)
                # log(-)=nan after center '((f1 - f2) / f1)**(1/{})'.format(
                # scale),
                # Widens wide 'where(f2 < f1, ((f1 - f2) / f1)**(1/{}),
                # 0)'.format(scale),

                # Hard to interpret # ((f1 - f2) / f1)^std(ei)
                # log(-)=nan after center '((f1 - f2) / f1)**(((f1 - f2) /
                # f1).std())',
                # Hard to interpret 'where(f2 < f1, ((f1 - f2) / f1)**(((f1 -
                #  f2) / f1).std()), 0) ',
                # Spikes to 0 after center 'where(f2 < f1, log( ((f1 - f2) /
                # f1)**(((f1 - f2) / f1).std()) ), 0) ',

                # Hard to interpret # ((f1 - f2) / f1)^(1/std(ei))
                # log(-)=nan after center '((f1 - f2) / f1)**(1/((f1 - f2) /
                # f1).std())',
                # Hard to interpret (best during entropy test)  'where(f2 <
                # f1, ((f1 - f2) / f1)**(1/((f1 - f2) / f1).std()), 0) ',
                # Same as just log 'where(f2 < f1, log( ((f1 - f2) / f1)**(
                # 1/((f1 - f2) / f1).std()) ), 0) ',
            ]
            eis = []

            # Plot each function
            for j, f in enumerate(functions):
                figure(figsize=figure_size_)

                # Compute essentiality index
                ei = _compute_essentiality_index(skew_t_pdf, skew_t_pdf_r, f,
                                                 ['+', '-'][shape > 0],
                                                 grids[1] - grids[0])

                c = CMAP_CATEGORICAL_TAB20(j / len(functions))
                eis.append((ei, c))

                plot(grids, ei, color=c, **line_kwargs)
                decorate(title=f)

            # Plot all functions
            figure(figsize=figure_size_)
            distplot(
                f_v,
                bins=n_bins,
                kde=False,
                norm_hist=True,
                hist_kws=dict(linewidth=0.92, color='#070707', alpha=0.26))
            for ei_, c in eis:
                plot(
                    grids, (ei_ - ei_.min()) /
                    (ei_.max() - ei_.min()) * skew_t_pdf.max(),
                    color=c,
                    linewidth=line_kwargs['linewidth'])
            decorate(title=f_i)

        # ==================================================================
        # Plot essentiality index (#FC154F)
        # ==================================================================
        ei = _compute_essentiality_index(
            skew_t_pdf, skew_t_pdf_r,
            'where(f2 < f1, ((f1 - f2) / f1)**{}, 0)'.format(scale),
            ['+', '-'][shape > 0], grids[1] - grids[0])
        ax_graph.plot(
            grids, (ei - ei.min()) / (ei.max() - ei.min()) * skew_t_pdf.max(),
            color='#FC154F',
            **line_kwargs)

        # ==================================================================
        # Save
        # ==================================================================
        if directory_path:
            save_plot(
                join(directory_path, 'essentiality_plot', '{}.png'.format(
                    f_i)))

        if show_plot:
            show()

        close()


def make_essentiality_matrix(feature_x_sample,
                             feature_x_fit,
                             n_grids=3000,
                             function='scaled_fractional_difference',
                             factor=1):
    """

    Arguments:
        feature_x_sample: DataFrame; (n_feature, n_sample)
        feature_x_fit: DataFrame;
        n_grids: int;
        function: str;
        factor: number;
    Returns:
        DataFrame; (n_feature, n_sample)
    """

    print('\tApplying {} to each feature ...'.format(function))

    empty_ = empty(feature_x_sample.shape)

    skew_t = ACSkewT_gen()

    for i, (f_i, f_v) in enumerate(feature_x_sample.iterrows()):

        # Build skew-t PDF
        grids = linspace(f_v.min(), f_v.max(), n_grids)
        n, df, shape, location, scale = feature_x_fit.ix[i, :]
        skew_t_pdf = skew_t.pdf(grids, df, shape, loc=location, scale=scale)

        # Build reflected skew-t PDF
        skew_t_pdf_r = skew_t.pdf(
            define_x_coordinates_for_reflection(skew_t_pdf, grids),
            df,
            shape,
            loc=location,
            scale=scale)

        # Set up function
        if function.startswith('scaled_fractional_difference'):
            function = 'where(f2 < f1, ((f1 - f2) / f1)**{}, 0)'.format(scale)

        ei = _compute_essentiality_index(skew_t_pdf, skew_t_pdf_r, function,
                                         ['+',
                                          '-'][shape > 0], grids[1] - grids[0])

        ei = normalize_1d_array(ei, '0-1')

        empty_[i, :] = ei[[argmin(abs(grids - x))
                           for x in asarray(f_v)]] * sign(shape) * factor

    return DataFrame(
        empty_, index=feature_x_sample.index, columns=feature_x_sample.columns)


def _compute_essentiality_index(f1,
                                f2,
                                function,
                                area_direction=None,
                                delta=None):
    """
    Make a function from f1 and f2.
    Arguments:
        f1: array; function on the top
        f2: array; function at the bottom
        area_direction: str; {'+', '-'}
        function: str; ei = eval(function)
    Returns:
        array; ei
    """

    if 'area' in function:  # Compute cumulative area

        # Compute delta area
        darea1 = f1 / f1.sum() * delta
        darea2 = f2 / f2.sum() * delta

        # Compute cumulative area
        if area_direction == '+':  # Forward
            carea1 = cumsum(darea1)
            carea2 = cumsum(darea2)

        elif area_direction == '-':  # Reverse
            carea1 = cumsum(darea1[::-1])[::-1]
            carea2 = cumsum(darea2[::-1])[::-1]

        else:
            raise ValueError(
                'Unknown area_direction: {}.'.format(area_direction))

    # Compute essentiality index
    dummy = log
    dummy = where
    dummy = carea1
    dummy = carea2
    return eval(function)
