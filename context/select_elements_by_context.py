def select_elements_by_context(
        context_matrix,
        select_context,
        n_top=None,
        select_automatically=False,
):

    if select_context not in (
            'negative',
            'positive',
    ):

        raise ValueError('Unknown select_context: {}.'.format(select_context))

    if select_context == 'negative':

        element_context = context_matrix[context_matrix < 0].sum(axis=1)

    elif select_context == 'positive':

        element_context = context_matrix[0 < context_matrix].sum(axis=1)

    element_context_absolute = element_context.abs()

    if n_top is not None:

        print('Selecting top {} elements ...'.format(n_top))

        elements = element_context_absolute.sort_values()[-n_top:].index

    elif select_automatically:

        print('Selecting elements automatcally ...')

        elements = element_context_absolute[
            element_context_absolute.std() < element_context_absolute].index

    return elements
