from numpy import ceil, sqrt
from scipy.stats import scoreatpercentile


def compute_n_histogram_bin_using_freedman_diaconis(array_1d):
    """
    Compute number of histotgram bins using Freedman-Diaconis.
        (http://stats.stackexchange.com/questions/798/)
    Arguments:
        array_1d (nd_array):
    Returns:
        int:
    """

    if array_1d.size < 2:
        return 1

    q1 = scoreatpercentile(array_1d, 25)
    q3 = scoreatpercentile(array_1d, 75)
    iqr = q3 - q1

    width = 2 * iqr / (array_1d.size**(1 / 3))
    if width == 0:
        return int(sqrt(array_1d.size))
    else:
        return int(ceil((array_1d.max() - array_1d.min()) / width))
