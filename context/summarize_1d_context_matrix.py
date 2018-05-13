def summarize_1d_context_matrix(_1d_context_matrix, select_context):

    if select_context not in ('negative', 'positive', 'both'):

        raise ValueError('Unknown select_context: {}.'.format(select_context))

    if select_context == 'negative':

        context_summary = _1d_context_matrix[_1d_context_matrix < 0].sum(
            axis=1)

    elif select_context == 'positive':

        context_summary = _1d_context_matrix[0 < _1d_context_matrix].sum(
            axis=1)

    elif select_context == 'both':

        context_summary = _1d_context_matrix.abs().sum(axis=1)

    return context_summary
