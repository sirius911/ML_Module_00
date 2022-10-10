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
    try:
        return((y_hat - y) * (y_hat - y))
    except Exception as e:
        print (e)
        return None
            
def loss_(y, y_hat):
    """
    Description: Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return loss_elem_(y, y_hat).sum() / (2 * len(y))
    except Exception as e:
        print (e)
        return None
