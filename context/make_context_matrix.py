from numpy import full, nan
from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .compute_context import compute_context
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def make_context_matrix(
        df,
        n_job=1,
        skew_t_pdf_fit_parameter=None,
        n_grid=1e3,
        degree_of_freedom_for_tail_reduction=1e8,
        multiply_distance_from_location=False,
        global_location=None,
        global_scale=None,
        global_degree_of_freedom=None,
        global_shape=None,
        directory_path=None,
):

    context_matrix = concat(
        multiprocess(
            _make_context_matrix,
            ((
                df_,
                skew_t_pdf_fit_parameter,
                n_grid,
                degree_of_freedom_for_tail_reduction,
                multiply_distance_from_location,
                global_location,
                global_scale,
                global_degree_of_freedom,
                global_shape,
            ) for df_ in split_df(
                df,
                0,
                min(
                    df.shape[0],
                    n_job,
                ),
            )),
            n_job,
        ))

    if directory_path is not None:

        establish_path(
            directory_path,
            'directory',
        )

        context_matrix.to_csv(
            '{}/context_matrix.tsv'.format(directory_path),
            sep='\t',
        )

    return context_matrix


def _make_context_matrix(
        df,
        skew_t_pdf_fit_parameter,
        n_grid,
        degree_of_freedom_for_tail_reduction,
        multiply_distance_from_location,
        global_location,
        global_scale,
        global_degree_of_freedom,
        global_shape,
):

    skew_t_model = ACSkewT_gen()

    context_matrix = full(
        df.shape,
        nan,
    )

    n = df.shape[0]

    n_per_print = max(
        1,
        n // 10,
    )

    for i, (
            index,
            series,
    ) in enumerate(df.iterrows()):

        if i % n_per_print == 0:

            print('({}/{}) {} ...'.format(
                i + 1,
                n,
                index,
            ))

        if skew_t_pdf_fit_parameter is None:

            location = scale = degree_of_freedom = shape = None

        else:

            location, scale, degree_of_freedom, shape = skew_t_pdf_fit_parameter.loc[
                index, [
                    'Location',
                    'Scale',
                    'Degree of Freedom',
                    'Shape',
                ]]

        context_matrix[i] = compute_context(
            series.values,
            skew_t_model=skew_t_model,
            location=location,
            scale=scale,
            degree_of_freedom=degree_of_freedom,
            shape=shape,
            n_grid=n_grid,
            degree_of_freedom_for_tail_reduction=
            degree_of_freedom_for_tail_reduction,
            multiply_distance_from_location=multiply_distance_from_location,
            global_location=global_location,
            global_scale=global_scale,
            global_degree_of_freedom=global_degree_of_freedom,
            global_shape=global_shape,
        )['context_indices_like_array']

    return DataFrame(
        context_matrix,
        index=df.index,
        columns=df.columns,
    )
