from numpy import sign


def summarize_context_indices(
        context_indices,
        summarize_context_indices_method='summarize_only_shape_side',
        shape=None):
    """
    Summarize context indices.
    Arguments:
        context_indices (array): (n)
        summarize_context_indices_method (str): 'summarize_only_shape_side'
        shape (float):
    Returns:
        float: context summary
    """

    if summarize_context_indices_method == 'summarize_only_shape_side':
        return ((sign(context_indices) == sign(shape)) * context_indices).sum()
    else:
        return context_indices.sum()
