import numpy as np

Q = np.array([[1, -4, 6, -4, 0],
              [0, 1, -4, 6, -4],
              [1, 2, -8, 0, 0],
              [0, 1, 2, -8, 0],
              [0, 0, 1, 2, -8]])

print("det(Q): ", np.linalg.det(Q))


Q1 = np.array([[-4, 6, -4, 0],
              [1, -4, 6, -4],
              [2, -8, 0, 0],
              [1, 2, -8, 0]])

Q2 = np.array([[1, 6, -4, 0],
              [0, -4, 6, -4],
              [1, -8, 0, 0],
              [0, 2, -8, 0]])

print("Common root: ", -1.0 * np.linalg.det(Q1) / np.linalg.det(Q2))
              