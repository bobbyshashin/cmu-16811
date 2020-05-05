import numpy as np
from math import *
import matplotlib.pyplot as plt

paths = np.loadtxt('paths.txt')
num_points = paths.shape[1]

start = np.zeros((2, 1))
# start[0] = 0.8
# start[1] = 1.8

# start[0] = 2.2
# start[1] = 1.0

start[0] = 2.7
start[1] = 1.4

# only choose paths that are on the same side of the fire ring
# check with the point in the middle of each path
candidate_paths = np.empty((0, num_points))

for i in range(0, paths.shape[0], 2):
    if (start[0] < start[1]) == (paths[i][num_points//2] < paths[i+1][num_points//2]):
        candidate_paths = np.append(candidate_paths, paths[i, :].reshape(-1, num_points), axis=0)
        candidate_paths = np.append(candidate_paths, paths[i+1, :].reshape(-1, num_points), axis=0)

print(candidate_paths.shape)
num_candidate_paths = candidate_paths.shape[0]

# Function to check whether point p is inside the triangle formed by p1, p2 and p3
def is_inside_triangle(x, y, x1, y1, x2, y2, x3, y3):
    c1 = (x2-x1) * (y-y1) - (y2-y1) * (x-x1)
    c2 = (x3-x2) * (y-y2) - (y3-y2) * (x-x2)
    c3 = (x1-x3) * (y-y3) - (y1-y3) * (x-x3)
    if (c1<0 and c2<0 and c3<0) or (c1>0 and c2>0 and c3>0):
        return True
    else:
        return False

def getDistSquare(x, y, x1, y1, x2, y2, x3, y3):
    return (x1-x)**2 + (y1-y)**2 + (x2-x)**2 + (y2-y)**2 + (x3-x)**2 + (y3-y)**2

def getWeight(x, y, x1, y1, x2, y2, x3, y3):
    c1 = ((x-x3)*(y2-y3) - (y-y3)*(x2-x3)) / ((x1-x3)*(y2-y3) - (y1-y3)*(x2-x3))
    c2 = ((x-x3)*(y1-y3) - (y-y3)*(x1-x3)) / ((y1-y3)*(x2-x3) - (x1-x3)*(y2-y3))
    c3 = 1 - c1 - c2

    return c1, c2, c3

start_points_x = []
start_points_y = []

candidate = None
min_dist = float('inf')


for i in range(0, num_candidate_paths, 2):
    start_points_x.append(candidate_paths[i][0])
    start_points_y.append(candidate_paths[i+1][0])
    for j in range(i, num_candidate_paths, 2):
        for k in range(j, num_candidate_paths, 2):
            if j < i+2 or k < j+2:
                pass
            else:
                inside = is_inside_triangle(start[0], start[1], candidate_paths[i, 0], candidate_paths[i+1, 0], candidate_paths[j, 0], candidate_paths[j+1, 0], candidate_paths[k, 0], candidate_paths[k+1, 0])
                if inside:
                    dist = getDistSquare(start[0], start[1], candidate_paths[i, 0], candidate_paths[i+1, 0], candidate_paths[j, 0], candidate_paths[j+1, 0], candidate_paths[k, 0], candidate_paths[k+1, 0])
                    if dist < min_dist:
                        candidate = np.array([i, j, k])
                        min_dist = dist

# print(candidate)

p1x = candidate_paths[candidate[0]]
p1y = candidate_paths[candidate[0]+1]
p2x = candidate_paths[candidate[1]]
p2y = candidate_paths[candidate[1]+1]
p3x = candidate_paths[candidate[2]]
p3y = candidate_paths[candidate[2]+1]

c1, c2, c3 = getWeight(start[0], start[1], p1x[0], p1y[0], p2x[0], p2y[0], p3x[0], p3y[0])
print("Weights: " + str(c1) + ", " + str(c2) + ", " + str(c3))

interpolated_path_x = np.zeros(num_points)
interpolated_path_y = np.zeros(num_points)

for i in range(num_points):
    interpolated_path_x[i] = c1*p1x[i] + c2*p2x[i] + c3*p3x[i]
    interpolated_path_y[i] = c1*p1y[i] + c2*p2y[i] + c3*p3y[i]


# Plot stuff
# plt.scatter(start_points_x, start_points_y, s=1.5, c='gray')
plt.scatter(start[0], start[1], s=15, c='gray')
plt.scatter(p1x[0], p1y[0], s=3*np.pi, c='b')
plt.scatter(p2x[0], p2y[0], s=3*np.pi, c='g')
plt.scatter(p3x[0], p3y[0], s=3*np.pi, c='r')

plt.plot(p1x, p1y, c='b', alpha=0.5, linewidth=2)
plt.plot(p2x, p2y, c='g', alpha=0.5, linewidth=2)
plt.plot(p3x, p3y, c='r', alpha=0.5, linewidth=2)
plt.plot(interpolated_path_x, interpolated_path_y, c='gray', alpha=1.0, linewidth=2)

ring_of_fire = plt.Circle((5, 5), 1.5, color='r', fill=False)

ax = plt.gca()
# ax.cla() # clear things for fresh plot

# change default range so that new circles will work
ax.set_xlim((0, 12))
ax.set_ylim((0, 12))


ax.add_artist(ring_of_fire)

plt.show()