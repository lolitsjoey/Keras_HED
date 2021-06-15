import random
from BatAlgorithm import *
import matplotlib.pyplot as plt


def Fun(D, sol):
    val = 0.0
    for i in range(D-1):
        val = val + (1-sol[i]**2)**2 + 100*(sol[i+1] - sol[i]**2)**2
    return val

# For reproducive results
#random.seed(5)
iters = 100
crit = 9999999
for i in range(iters):
    Algorithm = BatAlgorithm(10, 200, 1000, 0.5, 0.5, 0.0, 2.0, -2.048, 2.048, Fun)
    Algorithm.move_bat()

    x_coords = []
    for k in range(iters):
        sol = Algorithm.Sol[k]
        D = 10
        val = Fun(D, sol)
        x_coords.append(val)
    crit_compare = min(x_coords)
    if crit_compare < crit:
        crit = crit_compare
        best_coords = x_coords
plt.plot(best_coords)
plt.show()
