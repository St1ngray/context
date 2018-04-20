from os.path import join

from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .fit_skew_t_pdf import fit_skew_t_pdf
from .support.support.df import split_df
from .support.support.multiprocess import multiprocess
from .support.support.path import establish_path


def fit_skew_t_pdfs(matrix, n_job=1, directory_path=None):

    skew_t_pdf_fit_parameter = concat(
        multiprocess(_fit_skew_t_pdfs,
                     ((matrix_, ) for matrix_ in split_df(matrix, n_job)),
                     n_job))

    if directory_path:
        establish_path(directory_path, 'directory')

        skew_t_pdf_fit_parameter.to_csv(
            join(directory_path, 'skew_t_pdf_fit_parameter.tsv'), sep='\t')

    return skew_t_pdf_fit_parameter


def _fit_skew_t_pdfs(matrix):

    skew_t_model = ACSkewT_gen()

    skew_t_pdf_fit_parameter = DataFrame(
        index=matrix.index,
        columns=('N', 'Location', 'Scale', 'Degree of Freedom', 'Shape'),
        dtype=float)

    n = matrix.shape[0]
    n_per_print = max(n // 10, 1)

    for i, (index, vector) in enumerate(matrix.iterrows()):

        if i % n_per_print == 0:
            print('({}/{}) {} ...'.format(i + 1, n, index))

        skew_t_pdf_fit_parameter.loc[index] = fit_skew_t_pdf(
            vector, skew_t_model=skew_t_model)

    return skew_t_pdf_fit_parameter
