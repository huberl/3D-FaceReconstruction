import numpy as np


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
    if value == 1:
        extender = np.ones((array.shape[0], 1))
    elif value == 0:
        extender = np.zeros((array.shape[0], 1))
    else:
        extender = np.array([[value]])
        extender = extender.repeat(array.shape[0], axis=0)
    return np.hstack((array, extender))
