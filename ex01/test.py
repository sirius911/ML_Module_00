import numpy as np
import TinyStatistician as ts

a = [1, 42, 300, 10, 59]
tstat = ts.TinyStatistician()
# print(tstat.mean(a))

# print(tstat.mean("dfdf"))
# print(tstat.mean(np.array([[1,2,3],[2,3,4],[3,4,5.0]])))
# print(tstat.median(a))
# print(tstat.quartile(a))
# print(tstat.var(a))
# print(tstat.std(a))
# b = np.array([1, 2, 9])
# print(f"pour b = {b}, moy = {tstat.mean(b)} \
# var = {tstat.var(b)}, std = {tstat.std(b)}")

# print(tstat.percentile(a, 10))
# print(tstat.percentile(a, 15))
# print(tstat.percentile(a, 20))

data = [42, 7, 69, 18, 352, 3, 650, 754, 438, 2659]
epsilon = 1e-5
err = "Error, grade 0 :("

assert abs(tstat.mean(data) - 499.2) < epsilon, err
assert abs(tstat.median(data) - 210.5) < epsilon, err

quartile = tstat.quartile(data)
assert abs(quartile[0] - 18) < epsilon, err
assert abs(quartile[1] - 650) < epsilon, err