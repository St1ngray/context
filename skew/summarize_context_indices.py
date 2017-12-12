from numpy import sign


def summarize_context_indices(context_indices, shape):
    """
    Summarize context indices.
    Arguments:
        context_indices (array): (n)
        shape (float):
    Returns:
        float: context summary
    """

    return ((sign(context_indices) == sign(shape)) * context_indices).sum()
