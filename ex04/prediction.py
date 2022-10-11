import numpy as np

def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if x.shape[-1] == 0:
        return None
    if isinstance(x, np.ndarray):
        return (np.c_[np.ones(x.shape[0]), x])

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    # control args
    if not isinstance(x,np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) > 1 and x.shape[1] != 1:
        return None
    if (len(theta.shape) > 1 and theta.shape != (2, 1)) or (len(theta.shape) == 1 and theta.shape[0] != 2):
        return None
    if x.shape[-1] == 0 or theta.shape[-1] == 0:
        return None

    x_1 = add_intercept(x)
    return x_1.dot(theta)
