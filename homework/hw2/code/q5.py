
from math import *
import numpy as np

def orig_f(x):
    return (1.0 + 0.0j) * x**3 - (5.0 + 0.0j) * x**2 + (11.0 + 0.0j) * x - 15.0

# The function that we want to find roots for
def deflated_f(f, r1):
    def df(x):
        return f(x) / (x - r1)
    return df

# Given three initial guesses: x0, x1, x2
def Muller(f, x0, x1, x2):

    f0 = f(x0)
    f1 = f(x1)
    f2 = f(x2)

    f01 = (f1 - f0) / (x1 - x0) 

    f12 = (f2 - f1) / (x2 - x1) 

    f012 = (f12 - f01) / (x2 - x1)

    a = f012
    b = a * (x2 - x1) + f12

    c = np.sqrt(b * b - 4 * a * f2)

    d1 = b + c
    d2 = b - c

    if abs(d1) > abs(d2):
        d = d1
    else:
        d = d2

    x3 = x2 - 2.0 * f2 / d

    err = abs(x3 - x2)

    return x3, err

num_roots = 3
num_iter = 100
error_thres = 1e-5
f = orig_f

for n in range(num_roots):
    x = np.array([3+0j, 4+0j, 5+0j])

    for i in range(num_iter):
        r, err = Muller(f, x[-3], x[-2], x[-1])
        x = np.append(x, r)
        if err < error_thres:
            break

    r1 = x[-1]
    f = deflated_f(f, r1)
    print("The ", n+1, "th root is: ", r1)
