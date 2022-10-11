import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from other_losses import mae_, mse_, r2score_, rmse_

green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'
yellow = '\033[33m'

def okko(resultA, resultB):
    """
    return Ok or KO if resultA equal or not equal with resultB
    """
    if resultA == resultB:
        return (green + "OK" + reset)
    else:
        return (red + "KO" + reset)

x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])
# Mean squared error
slk = mean_squared_error(x,y)
ft =mse_(x,y)
print(f"mse = {ft} -> {okko(ft, slk)}")

# Root mean squared error
slk = sqrt(mean_squared_error(x,y))
ft = rmse_(x,y)
print(f"rmse = {ft} -> {okko(ft, slk)}")

# Mean absolute error
slk = mean_absolute_error(x,y)
ft = mae_(x, y)
print(f"mae = {ft} -> {okko(ft, slk)}")

# R2-score
slk = r2_score(x,y)
ft = r2score_(x,y)
print(f"r2score = {ft} -> {okko(ft, slk)}")
