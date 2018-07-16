from .summarize_1d_context_matrix import summarize_1d_context_matrix


def _select_elements_by_context_summary(
        _1d_context_matrix,
        select_context,
        n_top=None,
        select_automatically=False,
):

    absolute_context_summary = summarize_1d_context_matrix(
        _1d_context_matrix,
        select_context,
    ).abs()

    if n_top is not None:

        print('Selecting top {} elements ...'.format(n_top))

        elements = absolute_context_summary.sort_values()[-n_top:].index

    elif select_automatically:

        print('Selecting elements automatcally ...')

        elements = absolute_context_summary[
            absolute_context_summary.std() < absolute_context_summary].index

    else:

        print('Selecting all elements ...')

        elements = absolute_context_summary.index

    return elements
