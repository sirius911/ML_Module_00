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
    y = np.zeros((m,1))
    for i, xi in enumerate(x):
        y[i][0] = theta[0] + theta[1] * xi
    return y

def loss_elem_(y, y_hat):
    """
    Description:  Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray)\
         or y.shape[-1] == 0 or y_hat.shape[-1] == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    try:
        return((y_hat - y) * (y_hat - y))
    except Exception:
        return None

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
    print(f"x =\n{x}")
    print(f"Predictions : \n{prediction}")
    
    if prediction is not None:
        plt.scatter(x, y, c='blue')
        plt.plot(x, prediction, color='black')
        p1 = [x[3],y[3]]
        p2 = [x[3],prediction[3][0]]
        print(p1,p2)
        x__, y__ = [p1[0], p2[0]], [p1[1], p2[1]]
        print(x__,y__)
        plt.plot(x__,y__)
        # for x_,y_hat in zip(x,prediction):
        #     plt.axvline(x_, ymin=0.0,ymax=.5)
        plt.show()