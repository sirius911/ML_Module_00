import numpy as np
from TinyStatistician import TinyStatistician

a = [1, 42, 300, 10, 59]
print(TinyStatistician().mean(a))

print(TinyStatistician().mean("dfdf"))
print(TinyStatistician().mean(np.array([[1,2,3],[2,3,4],[3,4,5.0]])))
print(TinyStatistician().median(a))
print(TinyStatistician().quartiles(a))
print(TinyStatistician().var(a))
print(TinyStatistician().std(a))
b = np.array([1, 2, 9])
print(f"pour b = {b}, moy = {TinyStatistician().mean(b)} \
var = {TinyStatistician().var(b)}, std = {TinyStatistician().std(b)}")

print(TinyStatistician().percentile(a, 10))
print(TinyStatistician().percentile(a, 15))
print(TinyStatistician().percentile(a, 20))
