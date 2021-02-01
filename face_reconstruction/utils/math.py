import numpy as np
from scipy.spatial.distance import cdist, euclidean


def add_row(array: np.ndarray, value):
    """
    Adds a row at the bottom filled with `value` to the given (2d) array.

    Parameters
    ----------
        array: the array to add a row to
        value: the new row will be filled entirely with this value

    Returns
    -------
        the array with an additional row at the bottom
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if value == 1:
        extender = np.ones((1, array.shape[1]))
    elif value == 0:
        extender = np.zeros((1, array.shape[1]))
    else:
        extender = np.array([[value]])
        extender = extender.repeat(array.shape[1], axis=1)
    return np.vstack((array, extender))


def add_column(array: np.ndarray, value):
    """
    Adds a column at the right filled with `value` to the given (2d) array.

    Parameters
    ----------
        array: the array to add a column to
        value: the new column will be filled entirely with this value

    Returns
    -------
        the array with an additional column at the right
    """
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    if value == 1:
        extender = np.ones((array.shape[0], 1))
    elif value == 0:
        extender = np.zeros((array.shape[0], 1))
    else:
        extender = np.array([[value]])
        extender = extender.repeat(array.shape[0], axis=0)
    return np.hstack((array, extender))


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
