def _make_1d_signal_matrix(
        _1d_context_matrix,
        select_context,
):

    _1d_signal_matrix = _1d_context_matrix.copy()

    if select_context == 'negative':

        _1d_signal_matrix[0 < _1d_signal_matrix] = 0

        _1d_signal_matrix *= -1

    elif select_context == 'positive':

        _1d_signal_matrix[_1d_signal_matrix < 0] = 0

    return _1d_signal_matrix
