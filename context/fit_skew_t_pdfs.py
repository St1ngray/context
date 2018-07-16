from os.path import join

from numpy import full, nan
from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_skew_t_pdf import fit_skew_t_pdf
from .nd_array.nd_array.check_nd_array_for_bad_value import \
    check_nd_array_for_bad_value
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def fit_skew_t_pdfs(
        df,
        n_job=1,
        directory_path=None,
):

    skew_t_pdf_fit_parameter = concat(
        multiprocess(
            _fit_skew_t_pdfs,
            ((df_, ) for df_ in split_df(
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

        skew_t_pdf_fit_parameter.to_csv(
            join(
                directory_path,
                'skew_t_pdf_fit_parameter.tsv',
            ),
            sep='\t',
        )

    return skew_t_pdf_fit_parameter


def _fit_skew_t_pdfs(df):

    skew_t_model = ACSkewT_gen()

    skew_t_pdf_fit_parameter = full(
        (
            df.shape[0],
            5,
        ),
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

        _1d_array = series.values

        skew_t_pdf_fit_parameter[i] = fit_skew_t_pdf(
            _1d_array[~check_nd_array_for_bad_value(
                _1d_array,
                raise_for_bad_value=False,
            )],
            skew_t_model=skew_t_model,
        )

    return DataFrame(
        skew_t_pdf_fit_parameter,
        index=df.index,
        columns=(
            'N',
            'Location',
            'Scale',
            'Degree of Freedom',
            'Shape',
        ),
    )
