from math import *
import numpy as np

def divided_diff(f, known_x, x):
    n = known_x.shape[0]
    F = np.zeros((n, n))
    A = np.zeros(n)

    for i in range(n):
        F[0, i] = f(known_x[i])

    for i in range(n):
        if i != 0:
            for j in range(n-i):
                F[i, j] = (F[i-1, j] - F[i-1, j+1]) / (known_x[j] - known_x[i+j])
        
        A[i] = F[i, 0]

    result = 0.0
    for i in range(n):
        m = 1.0
        for j in range(i):
            m *= (x - known_x[j])
        result += A[i] * m

    return result

# (b)
def f1(x):
    return exp(-1.0 * x)

x = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
query = 1.0 / 3.0
print("Interpolated value at x =", query, "is:", divided_diff(f1, x, query))

actual = f1(query)

print("Actual value at x =", query, "is:", actual)

# (c)
def f2(x):
    return 1.0 / (1.0 + 16.0 * x * x)

def construct_x(n):
    x = np.zeros(n+1)
    for i in range(n+1):
        x[i] = i * 2.0 / n - 1.0
    return x

num_points = [2, 4, 40]
query = 0.05

for n in num_points:

    x = construct_x(n)

    result = divided_diff(f2, x, query)

    print("Interpolated value at x =", query, " with n =", n, "points is:", result)

actual = f2(query)
print("Actual value at x =", query, "is:", actual)

# (d)
def f3(x):
    return 1.0 / (1.0 + 16.0 * x * x)
    
num_points = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]

for n in num_points:

    x = construct_x(n)
    # print(x)
    err_max = float('-inf')
    xi = -1.0
    while xi <= 1.0:
        result = divided_diff(f3, x, xi)
        err = abs(f3(xi) - result)
        # print("x:", xi)
        # print("err:",err)
        if err > err_max:
            err_max = err
        xi += 0.0001

    print("Max error for n =", n, "is:", err_max)
