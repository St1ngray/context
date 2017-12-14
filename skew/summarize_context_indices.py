from numpy import sign


def summarize_context_indices(context_indices, shape=None):
    """
    Summarize context indices.
    Arguments:
        context_indices (array): (n)
        shape (float):
    Returns:
        float: context summary
    """

    if shape is None:
        return context_indices.sum()
    else:
        return ((sign(context_indices) == sign(shape)) * context_indices).sum()
