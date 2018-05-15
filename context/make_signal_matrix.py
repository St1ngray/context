def make_signal_matrix(_1d_context_matrix, select_context):

    if select_context not in ('negative', 'positive', 'both'):

        raise ValueError('Unknown select_context: {}.'.format(select_context))

    _1d_context_matrix = _1d_context_matrix.copy()

    if select_context == 'negative':

        _1d_context_matrix[0 < _1d_context_matrix] = 0

        _1d_context_matrix *= -1

    elif select_context == 'positive':

        _1d_context_matrix[_1d_context_matrix < 0] = 0

    elif select_context == 'both':

        raise NotImplementedError()

    return _1d_context_matrix
