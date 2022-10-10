import numpy as np

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
    m = x.shape[0]
    y = np.zeros((m,1))
    for i, xi in enumerate(x):
        y[i][0] = theta[0] + theta[1] * xi
    return y