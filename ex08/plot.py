import numpy as np
import matplotlib.pyplot as plt

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
    m = x.shape[0]
    # y = np.zeros((m,1))
    y = np.zeros(m)
    for i, xi in enumerate(x):
        y[i] = theta[0] + theta[1] * xi
    return y

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    prediction = predict_(x, theta)
    n = len(x)
    if prediction is not None:
        loss_elem = (prediction - y)*(prediction - y)
        # print(loss_elem)
        cost =loss_elem.sum() / len(y)
        plt.scatter(x, y, c='blue')
        plt.plot(x, prediction, color='red')
        for i in range(n):
            pA = [x[i],y[i]]
            pB = [x[i],prediction[i]]
            x__, y__ = [pA[0], pB[0]], [pA[1], pB[1]]
            plt.plot(x__, y__, 'r--')
        plt.title("Cost : "+str(cost))
        plt.show()