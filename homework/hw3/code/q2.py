import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt

fx = np.loadtxt('problem2.txt').reshape(-1, 1)
fx1 = fx[0:31]
fx2 = fx[31:]
print(fx2.shape)
x = np.zeros(101)
x1 = np.zeros(31)
x2 = np.zeros(70)

phi0 = np.zeros((31,1))
phi1 = np.zeros((31,1))
phi2 = np.zeros((31,1))
phi3 = np.zeros((31,1))
phi4 = np.zeros((31,1))

phi5 = np.zeros((70,1))
phi6 = np.zeros((70,1))
phi7 = np.zeros((70,1))
phi8 = np.zeros((70,1))
phi9 = np.zeros((70,1))
# phi9 = np.zeros((101,1))
# phi10 = np.zeros((101,1))
# phi11 = np.zeros((101,1))
# phi12 = np.zeros((101,1))
# phi13 = np.zeros((101,1))
# phi14 = np.zeros((101,1))

def f0(x):
    return 1.0
def f1(x):
    return x
def f2(x):
    return x * x
def f3(x):
    return pow(x, 3)
def f4(x):
    return pow(x, 4)
def f5(x):
    return sin(pi * x)
def f6(x):
    return cos(pi * x)
for i in range(101):
    x[i] = i / 10.0

for i in range(31):
    x1[i] = i / 10.0
    phi0[i] = f0(x1[i])
    phi1[i] = f1(x1[i])
    phi2[i] = f2(x1[i])
    phi3[i] = f3(x1[i])
    phi4[i] = f4(x1[i])

for i in range(70):
    x2[i] = i / 10.0 + 3.0
    phi5[i] = f0(x2[i])
    phi6[i] = f1(x2[i])
    phi7[i] = f2(x2[i])
    phi8[i] = f3(x2[i])
    phi9[i] = f4(x2[i])


A1 = np.concatenate((phi0, phi1, phi2, phi3, phi4), axis=1)
A2 = np.concatenate((phi5, phi6, phi7, phi8, phi9), axis=1)
# A = np.append(phi0, phi1, axis=1)
# A = np.append(A, phi2, axis=1)
# A = np.append(A, phi3, axis=1)

# print(A.shape)

U1, S1, V1 = np.linalg.svd(A1, full_matrices=False)
U2, S2, V2 = np.linalg.svd(A2, full_matrices=False)
# print(U.shape)
# print(S.shape)
# print(V.shape)


c1 = V1.T @ np.linalg.inv(np.diag(S1)) @ U1.T @ fx1
for i in range(c1.shape[0]):
    if abs(c1[i]) < 10e-3:
        c1[i] = 0
print(c1)

c2 = V2.T @ np.linalg.inv(np.diag(S2)) @ U2.T @ fx2
for i in range(c2.shape[0]):
    if abs(c2[i]) < 10e-3:
        c2[i] = 0
print(c2)
# err = 0
# for i in range(101):
#     est_x = c[0] * f0(x[i]) + c[1] * f1(x[i]) + c[2] * f2(x[i]) + c[3] * f3(x[i]) + c[4] * f4(x[i]) + c[5] * f5(x[i]) + c[6] * f6(x[i])
#     err += abs(fx[i] - est_x)

err1 = fx1 - A1 @ c1
err2 = fx2 - A2 @ c2
# print(np.sum(err) / err.shape[0])
# print(err)
print(np.mean(abs(err1), dtype=np.float32))
print(np.mean(abs(err2), dtype=np.float32))

# plt.plot(x, fx)
plt.plot(x1, fx1)
plt.plot(x2, fx2)
plt.plot(x1, A1 @ c1)
plt.plot(x2, A2 @ c2)
plt.show()