import numpy as np
from math import *

# less than the actual PI
PI_minus = 3.14
PI_plus = 3.1416
PI = PI_minus
def f(x):
    return 1.0 * x - tan(x)

def df(x):
    return  -1.0 * tan(x) * tan(x)

def Newton(x, err_thres):
    err = f(x)
    while (abs(err) > err_thres):
        x -= (f(x) / df(x))
        # print("x: ", x)
        
        err = f(x)
        # print("error: ", err)
    return x

x1 = 3*PI/2
x_midd = 4*PI/2
x2 = 5*PI/2

point = 7.0

# print("f(3pi/2):", f(x1))
# print("f(4pi/2):", f(x_midd))
# print("f(5pi/2):", f(x2))
# print("f(7):", f(point))

err_threshold = 1e-10

root1 = Newton(x1, err_threshold)
root2 = Newton(x2, err_threshold)

print("Root 1: ", root1)
print("Root 2: ", root2)



