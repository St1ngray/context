from statsmodels.sandbox.distributions.extras import ACSkewT_gen


def fit_skew_t_pdf(array_1d, skew_t_model=None):
    """
    Fit skew-t PDF.
    Arguments:
        array_1d (array): (n, )
        skew_t_model (statsmodels.sandbox.distributions.extras.ACSkewT_gen):
    Returns:
        int: n
        float: location
        float: scale
        float: df
        float: shape
    """

    if skew_t_model is None:
        skew_t_model = ACSkewT_gen()

    df, shape, location, scale = skew_t_model.fit(
        array_1d, loc=array_1d.mean(), scale=array_1d.std())

    return array_1d.size, location, scale, df, shape
