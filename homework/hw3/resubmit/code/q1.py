import numpy as np
import matplotlib.pyplot as plt

# def f(x):
#     return 1.0/3 + np.exp(x) - np.exp(-x)
# def g(x):
#     return 5.38726*x + 1.0/3
# x = np.zeros(600)
# fx = np.zeros(600)
# gx = np.zeros(600)

# max_err = -1
# for i in range(600):
#     x[i] = -3.0 + 0.01 * i
#     fx[i] = f(x[i])
#     gx[i] = g(x[i]) 
#     err = abs(fx[i] - gx[i])
#     if err > max_err:
#         max_err = err

# print("Max err", max_err)
# plt.plot(x, fx, label='f(x)')
# plt.plot(x, gx, label='g(x)')
# plt.legend()
# plt.show()
def f(x):
    return 1.0/3 + np.exp(x) - np.exp(-x)
def g(x):
    return 4.4856*x + 1.0/3
x = np.zeros(600)
fx = np.zeros(600)
gx = np.zeros(600)

max_err = -1
for i in range(600):
    x[i] = -3.0 + 0.01 * i
    fx[i] = f(x[i])
    gx[i] = g(x[i]) 
    err = abs(fx[i] - gx[i])
    if err > max_err:
        max_err = err

print("Max err", max_err)
plt.plot(x, fx, label='f(x)')
plt.plot(x, gx, label='g(x)')
plt.legend()
plt.show()