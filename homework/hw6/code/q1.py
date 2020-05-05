import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D    
import random

# randomly generate 1000 points in a 1000*1000 square area
num_points = 100
scale = 500
coords = [(random.random()-0.5, random.random()-0.5) for _ in range(num_points)]
coords = np.array(coords) * scale
# print(coords)

# 0: colinear
# 1: clockwise
# 2: counter-clockwise
def orientation(p, q, r):
    result = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if result == 0:
        return 0
    elif result > 0:
        return 1
    else:
        return 2

def findNextPoint(points, curr_point):
    n = points.shape[0]
    for i in range(n):
        if i == curr_point:
            continue
        
def findConvexHull(points):
    convex_hull = []
    start = np.argmin(points[:, 0])
    p = start
    q = None
    n = points.shape[0]

    # Convex hull should consists of at least 3 points
    if n < 3:
        return None
    
    while True:

        convex_hull.append(p)
        q = (p+1)%n

        for i in range(n):
            if orientation(points[p], points[i], points[q]) == 2:
                q = i
        # set q as the next point
        p = q

        # terminate once we come back to the start point
        # which means we already get a closed convex hull
        if p == start:
            break
        
    convex_hull.append(start)
    return convex_hull


convex_hull = findConvexHull(coords)
x = []
y = []
for id in convex_hull:
    x.append(coords[id, 0])
    y.append(coords[id, 1])

fig = plt.figure()
ax = fig.add_subplot(111)
line = Line2D(x, y, linewidth=2)
ax.add_line(line)
plt.plot(coords[:,0], coords[:,1], 'o', color='black')
plt.xlim(-scale*0.8, scale*0.8);
plt.ylim(-scale*0.8, scale*0.8);
plt.show()