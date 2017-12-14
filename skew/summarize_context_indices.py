from numpy import sign


def summarize_context_indices(
        context_indices,
        summarize_context_indices_method='summarize_shape_side',
        shape=None):
    """
    Summarize context indices.
    Arguments:
        context_indices (array): (n)
        summarize_context_indices_method (str): 'summarize_shape_side' |
            'summarize_both_side'
        shape (float):
    Returns:
        float: context summary
    """

    if summarize_context_indices_method == 'summarize_shape_side':
        return ((sign(context_indices) == sign(shape)) * context_indices).sum()

    elif summarize_context_indices_method == 'summarize_both_side':
        return context_indices.sum()

    else:
        raise ValueError('Unknown summarize_context_indices_method {}.'.format(
            summarize_context_indices_method))
