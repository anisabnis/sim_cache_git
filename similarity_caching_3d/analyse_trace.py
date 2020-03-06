import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import BoundaryNorm

v_id = str(sys.argv[1])

xs = defaultdict(list)
ys = defaultdict(list)

max_z = 0
max_x = 20
max_y = 20

#f = open("traces/trace_" + v_id + ".txt", "r")
f = open("new_traces/trace_" + str(v_id) + ".txt", "r")

for l in f:
    l = l.strip().split(" ")
    z = int(l[1])

    if z > max_z:
        max_z = z

    x = int(l[3])
    y = int(l[4])

    xs[z].append(x)
    ys[z].append(y)

arr_z = range(max_z)
arr_x = range(max_x)
arr_y = range(max_y)

intensity = np.zeros((max_z, max_x))
for z in arr_z:
    if z in xs:
        if type(xs[z]) is int:
            intensity[z][xs[z]] += 1
        else:
            for x in xs[z]:
                intensity[z][x] += 1

intensity = np.array(intensity)
arr_z = np.array(arr_z)
arr_x = np.array(arr_x)
arr_y = np.array(arr_y)

cmap = plt.get_cmap('hsv')
# Construct 2D histogram from data using the 'plasma' colormap
plt.pcolormesh(arr_x, arr_z, intensity, cmap=cmap)
plt.savefig("HeatMapTraceXs.png")


intensity = np.zeros((max_z, max_y))
for z in arr_z:
    if z in ys:
        if type(ys[z]) is int:
            intensity[z][ys[z]] += 1
        else:
            for y in ys[z]:
                intensity[z][y] += 1

intensity = np.array(intensity)
cmap = plt.get_cmap('hsv')
# Construct 2D histogram from data using the 'plasma' colormap
plt.pcolormesh(arr_y, arr_z, intensity, cmap=cmap)
plt.savefig("HeatMapTraceYs.png")
