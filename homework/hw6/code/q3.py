import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D    
import random

# Find the orientation relationship between 3 given points
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

    # Convex hull should consist of at least 3 points
    if n < 3:
        return None
    
    while True:
        convex_hull.append(p)
        q = (p+1)%n
        # find q as the "most counter-clockwise" point among all
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

# randomly generate 1000 points in a 1000*1000 square area
offsetX = [-600, 0, 0, 0, 600]
offsetY = [0, 0, 600, -600, 0]
colors = ['y', 'g', 'm', 'c', 'k']
num_points = 6
scale = 500
num_polygon = 5

fig = plt.figure()
ax = fig.add_subplot(111)

# initialize a pair of random start and goal for the shortest path algorithm
# start is bias sampled to the bottom left area
# goal is bias sampled to the upper right area
start = np.array((random.random()-1.5, random.random()-1.5)) * scale * 2
goal = np.array((random.random()+0.5, random.random()+0.5)) * scale * 2

# construct a robot with random shape near start
robot = [(random.random(), random.random()) for _ in range(10)]
flipped_robot = np.array(robot)
normalized_flipped_robot = -1.0 * flipped_robot
robot = np.array(robot) * 100 + start
flipped_robot = -1.0 * flipped_robot * 100 + start
robot = np.append(robot, start.reshape(1, 2), axis=0)
flipped_robot = np.append(flipped_robot, start.reshape(1, 2), axis=0)

robot_convex_hull = findConvexHull(robot)
flipped_robot_convex_hull = findConvexHull(flipped_robot)
convex_robot = []
convex_flipped_robot = []
for i in range(len(robot_convex_hull)):
    convex_robot.append(robot[robot_convex_hull[i]])
for i in range(len(flipped_robot_convex_hull)):
    convex_flipped_robot.append(flipped_robot[flipped_robot_convex_hull[i]])
convex_robot = np.array(convex_robot)
convex_flipped_robot = np.array(convex_flipped_robot)
plt.plot(convex_robot[:,0], convex_robot[:,1], 'o', color='r')
# plt.plot(convex_flipped_robot[:,0], convex_flipped_robot[:,1], 'o', color='b')
x = []
y = []
for id in robot_convex_hull:
    x.append(robot[id, 0])
    y.append(robot[id, 1])
line = Line2D(x, y, color='r', linewidth=3)
ax.add_line(line)

# x = []
# y = []
# for id in flipped_robot_convex_hull:
#     x.append(flipped_robot[id, 0])
#     y.append(flipped_robot[id, 1])
# line = Line2D(x, y, color='b', linewidth=3)
# ax.add_line(line)


# Construct polygons
point_set = []
convex_hulls = []
for n in range(num_polygon):
    coords = [(random.random()-0.5, random.random()-0.5) for _ in range(num_points)]
    coords = np.array(coords) * scale + np.array((offsetX[n], offsetY[n]))
    point_set.append(coords)

    convex_hull = findConvexHull(coords)
    convex_hulls.append(convex_hull)
    x = []
    y = []
    for id in convex_hull:
        x.append(coords[id, 0])
        y.append(coords[id, 1])


    line = Line2D(x, y, color=colors[n], linewidth=3)
    ax.add_line(line)
    # plt.plot(coords[:,0], coords[:,1], 'o', color=colors[n])

# for each set, remove points that does not on the edges
old_polygons = []
polygons = []
for i in range(len(point_set)):
    polygon = []
    for j in range(len(convex_hulls[i])):
        polygon.append(point_set[i][convex_hulls[i][j]])
    # print(polygon)
    old_polygons.append(np.array(polygon))

    for pid in range(len(polygon)):
        p = polygon[pid]
        for robot_v in range(normalized_flipped_robot.shape[0]):
            new_p = p + normalized_flipped_robot[robot_v] * 100

            polygon.append(new_p)
    polygon = np.array(polygon) 
    # plt.plot(polygon[:,0], polygon[:,1], 'o', color='b')
    ch = findConvexHull(np.array(polygon))
    new_polygon = []
    for j in range(len(ch)):
        new_polygon.append(polygon[ch[j]])

    x = []
    y = []
    for v in new_polygon:
        x.append(v[0])
        y.append(v[1])


    line = Line2D(x, y, color=colors[i], linewidth=3, linestyle='-.')
    ax.add_line(line)

    polygons.append(np.array(new_polygon))

# plt.xlim(-scale*3, scale*3)
# plt.ylim(-scale*3, scale*3)
# plt.show()

for n in range(len(polygons)):
    plt.plot(polygons[n][:,0], polygons[n][:,1], 'o', color=colors[n])


all_line_segments = []
for polygon in polygons:
    # print(polygon)
    for i in range(polygon.shape[0] - 1):
        line_segment = (polygon[i], polygon[i+1])
        all_line_segments.append(line_segment)
# for ls in all_line_segments:
#     print(ls)
def onSegment(p, q, r):
    return (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))

def intersect(p1, q1, p2, q2):
    # if the two segments share at least one common vertex, we consider it as not intersected
    if all(p1 == p2) or all(q1 == q2) or all(p1 == q2) or all(q1 == p2):
        return False
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if (o1 != o2) and (o3 != o4):
        return True
    # if one end is on the other segment, we does not consider as intersect

    # if o1 == 0 and onSegment(p1, p2, q1):
    #     return True
    # if o2 == 0 and onSegment(p1, q2, q1):
    #     return True
    # if o3 == 0 and onSegment(p2, p1, q2):
    #     return True
    # if o4 == 0 and onSegment(p2, q1, q2):
    #     return True

    return False

def isValidEdge(p1, p2, all_segments):
    for s in all_segments:
        if intersect(p1, p2, s[0], s[1]):
            return False
    return True
def hasEdge(all_edges, id1, id2):
    for edge in all_edges:
        if (edge[0] == id1) and (edge[1] == id2):
            return True
        if (edge[0] == id2) and (edge[1] == id1):
            return True
    return False

def addEdge(all_edges, id1, id2):
    if not hasEdge(all_edges, id1, id2):
        all_edges.append((id1, id2))
    # all_edges.append((id2, id1))


edges = []
start_id = (-1, -1)
goal_id = (-2, -2)

# construct visibility graph
# Note that for all polygon list, the last element is equal to the first element (the start is duplicated, to make it closed), so we -1 for indices below
for i in range(num_polygon):
    for j in range(len(polygons[i])-1):
        vertex = polygons[i][j]
        for k in range(num_polygon):
            # connect this vertex to all the vertices of other polygons
            if k != i:
                for l in range(len(polygons[k])-1):
                    if isValidEdge(vertex, polygons[k][l], all_line_segments):
                        addEdge(edges, (i, j), (k, l))
            # connect this vertex to its adjacent vertices of the same polygon
            else:
                f = (j+1) % (len(polygons[i])-1)
                addEdge(edges, (i, j), (i, f))
            
        # connect this vertex to start and goal
        if isValidEdge(vertex, start, all_line_segments):
            addEdge(edges, start_id, (i, j))
        if isValidEdge(vertex, goal, all_line_segments):
            addEdge(edges, goal_id, (i, j))

# connect start and goal, if possible
if isValidEdge(start, goal, all_line_segments):
    addEdge(edges, start_id, goal_id)

# plot visibility graph
for edge in edges:
    x = []
    y = []
    p1_id = edge[0]
    p2_id = edge[1]
    p1 = None
    p2 = None
    
    if p1_id == start_id:
        p1 = start
    elif p1_id == goal_id:
        p1 = goal
    else:
        p1 = polygons[p1_id[0]][p1_id[1]]

    if p2_id == start_id:
        p2 = start
    elif p2_id == goal_id:
        p2 = goal
    else:
        p2 = polygons[p2_id[0]][p2_id[1]]

    x.append(p1[0])
    y.append(p1[1])
    x.append(p2[0])
    y.append(p2[1])

    line = Line2D(x, y, color='r', alpha=0.2, linestyle='-.')
    ax.add_line(line)
    # print(key)

# dijkstra
def findSmallestCost(open_list):
    min_cost = float('inf')
    min_key = None
    for key in open_list.keys():
        if open_list[key] < min_cost:
            min_key = (key[0], key[1])
            min_cost = open_list[key]
    return min_key, min_cost

def euclideanDist(p1, p2):
    return np.linalg.norm(p1 - p2)

def findPath(parents, goal_id, start_id):
    parent_id = goal_id
    path = []
    while parent_id != start_id:
        path.append(parent_id)
        parent_id = parents[parent_id]
    path.append(start_id)
    path.reverse()
    return path

open_list = dict()
closed_list = set()
parents = dict()
open_list[start_id] = 0.0
path = None
# print(edges)
while bool(open_list):
    # get id with least cost
    # print(open_list)
    curr_id, curr_cost = findSmallestCost(open_list)
    # print("explore")
    if curr_id == goal_id:
        print("goal found!")
        path = findPath(parents, goal_id, start_id)
        break
    del open_list[curr_id]
    closed_list.add(curr_id)
    # expand
    for e in edges:
        new_id = None
        if curr_id == e[0]:
            new_id = (e[1][0], e[1][1])
        elif curr_id == e[1]:
            new_id = (e[0][0], e[0][1])

        if (new_id != None) and (new_id not in closed_list):
            p1 = None
            p2 = None
            if new_id == start_id:
                p1 = start
            elif new_id == goal_id:
                p1 = goal
            else:
                p1 = polygons[new_id[0]][new_id[1]]
            if curr_id == start_id:
                p2 = start
            elif curr_id == goal_id:
                p2 = goal
            else:
                p2 = polygons[curr_id[0]][curr_id[1]]
            new_cost = euclideanDist(p1, p2) + curr_cost

            # if already explored, check for smaller cost
            if new_id in open_list:
                prev_cost = open_list[new_id]
                # update cost and parent if this way has lower cost
                if new_cost < prev_cost:
                    open_list[new_id] = new_cost
                    parents[new_id] = (curr_id[0], curr_id[1])
                    # print("updated", new_cost)
            # newly explored, add to open list
            else:
                open_list[new_id] = new_cost
                parents[new_id] = (curr_id[0], curr_id[1])
# print(path)
# print path
robot_footprints = []
robot_shape = convex_robot - start
for i in range(len(path)-1):
    id1 = path[i]
    id2 = path[i+1]
    x = []
    y = []
    if id1 == start_id:
        p1 = start
    elif id1 == goal_id:
        p1 = goal
    else:
        p1 = polygons[id1[0]][id1[1]]

    if id2 == start_id:
        p2 = start
    elif id2 == goal_id:
        p2 = goal
    else:
        p2 = polygons[id2[0]][id2[1]]

    x.append(p1[0])
    y.append(p1[1])
    x.append(p2[0])
    y.append(p2[1])

    line = Line2D(x, y, color='b', linewidth=2)
    ax.add_line(line)

    footprint = []
    x = []
    y = []
    for v in range(robot_shape.shape[0]):
        new_p = p2 + robot_shape[v]
        footprint.append(new_p)
        
        x.append(new_p[0])
        y.append(new_p[1])

    line = Line2D(x, y, color='r', linewidth=3, linestyle='-.')
    ax.add_line(line)




# plot start and goal
plt.plot(start[0], start[1], 'o', color='r')
plt.plot(goal[0], goal[1], 'o', color='r')

plt.xlim(-scale*3, scale*3)
plt.ylim(-scale*3, scale*3)
plt.show()