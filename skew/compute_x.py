def _compute_essentiality_index(f1,
                                f2,
                                function,
                                area_direction=None,
                                delta=None):
    """
    Make a function from f1 and f2.
    Arguments:
        f1: array; function on the top
        f2: array; function at the bottom
        area_direction: str; {'+', '-'}
        function: str; ei = eval(function)
    Returns:
        array; ei
    """

    if 'area' in function:  # Compute cumulative area

        # Compute delta area
        darea1 = f1 / f1.sum() * delta
        darea2 = f2 / f2.sum() * delta

        # Compute cumulative area
        if area_direction == '+':  # Forward
            carea1 = cumsum(darea1)
            carea2 = cumsum(darea2)

        elif area_direction == '-':  # Reverse
            carea1 = cumsum(darea1[::-1])[::-1]
            carea2 = cumsum(darea2[::-1])[::-1]

        else:
            raise ValueError(
                'Unknown area_direction: {}.'.format(area_direction))

    # Compute essentiality index
    dummy = log
    dummy = where
    dummy = carea1
    dummy = carea2
    return eval(function)
