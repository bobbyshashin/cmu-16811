import numpy as np
from math import *
import matplotlib.pyplot as plt

step = -0.05
x = []

x0 = 2
y0 = sqrt(2)

for i in range(21):
    x.append(2 + i * step)
print(x)
x = np.array(x)

def real_y(x):
    return np.sqrt(2*x - 2)
def dy(x, y):
    return 1.0 / y

y_real = real_y(x)  

y_euler = np.zeros(x.shape)
y_euler[0] = y0

for i in range(x.shape[0]-1):
    y_euler[i+1] = y_euler[i] + dy(x[i], y_euler[i]) * step

# for i in range(x.shape[0]):
#     print(str(x[i]) + " & " + str(y_euler[i]) + " & " + str(y_real[i] - y_euler[i]) + " \ \ ")

# plt.plot(x, y_real, label='y(x)')
# plt.plot(x, y_euler, label='y_estimated')
# plt.legend()
# plt.show()

def f(y):
    return 1.0 / y

y_rk = np.zeros(x.shape)
y_rk[0] = y0

for i in range(x.shape[0]-1):
    k1 = f(y_rk[i])
    k2 = f(y_rk[i] + 0.5 * step * k1)
    k3 = f(y_rk[i] + 0.5 * step * k2)
    k4 = f(y_rk[i] + step * k3)
    y_rk[i+1] = y_rk[i] + 1.0 / 6 * (k1 + 2*k2 + 2*k3 + k4) * step

# for i in range(x.shape[0]):
#     print(str(x[i]) + " & " + str(y_rk[i]) + " & " + str(y_real[i] - y_rk[i]) + " \ \ ")

# plt.plot(x, y_real, label='y(x)')
# plt.plot(x, y_rk, label='y_estimated')    
# plt.legend()
# plt.show()

def g(x, y):
    return 1.0 / y

y_ab = np.zeros(x.shape)
y_ab[0] = y0

f1 = np.zeros(4)
f1[0] = g(2.15, 1.51657508881031)
f1[1] = g(2.10, 1.48323969741913)
f1[2] = g(2.05, 1.44913767461894)
f1[3] = g(2.00, 1.4142135623731)


for i in range(x.shape[0]-1):
    y_ab[i+1] = y_ab[i] + (step / 24.0) * (55.0 * f1[3] - 59.0 * f1[2] + 37.0 * f1[1] - 9.0 * f1[0]);
    f1[0] = f1[1]
    f1[1] = f1[2]
    f1[2] = f1[3]
    f1[3] = g(x[i+1], y_ab[i+1])

for i in range(x.shape[0]):
    print(str(x[i]) + " & " + str(y_ab[i]) + " & " + str(y_real[i] - y_ab[i]) + " \ \ ")

plt.plot(x, y_real, label='y(x)')
plt.plot(x, y_ab, label='y_estimated')
plt.legend()
plt.show()