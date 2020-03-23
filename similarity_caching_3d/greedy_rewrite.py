import sys
from collections import defaultdict
import bisect
import numpy as np
import matplotlib.pyplot as plt
import random

MISS = 8
APPROX = 1

TRACE = "new_traces/trace_14_Warship_shuffle1.txt"
CSIZE = int(sys.argv[1])

score_list = []
score_map = defaultdict(lambda : [])

neighbours = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]

## Cost benefit provided by the tile
C = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))

## Number of requests made for the tile
N = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))

## If the tile has been approximated
A = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))

## If the tile is in cache
I = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : 0)))

total_reqs = 0

f = open(TRACE, "r")

for l in f:
    l = l.strip().split(" ")
    x, y, z = int(l[3]), int(l[4]), int(l[1])
    C[x][y][z] += MISS
    N[x][y][z] += 1

    total_reqs += 1

    for n in neighbours:
        x_n = (x + n[0])%20
        y_n = (y + n[1])%20
        z_n = z
        C[x_n][y_n][z_n] += (MISS - APPROX)
        
f.close()

for x in C:
    for y in C[x]:
        for z in C[x][y]:
            score = C[x][y][z]
            bisect.insort(score_list, score)
            score_map[score].append([x, y, z])


def recompute_scores(x, y, z):
    cost = 0

    if I[x][y][z] == 1:
        return

    ## Remove from score_list and map
    init_cost = C[x][y][z]
    del score_list[bisect.bisect_left(score_list, init_cost)]
    score_map[init_cost] = [u for u in score_map[init_cost] if u != [x,y,z]]

    if A[x][y][z] != 1:
        cost += MISS * N[x][y][z]
    else:
        cost += APPROX * N[x][y][z]

    for n in neighbours:
        x1, y1, z1 = (x + n[0])%20, (y + n[1])%20, z
        if A[x1][y1][z1] != 1 and I[x1][y1][z1] != 1:
            cost += (MISS - APPROX) * N[x1][y1][z1]

    C[x][y][z] = cost

    # Add to score_list and map
    bisect.insort(score_list, cost)
    score_map[cost].append([x,y,z])

def costC():
    pass
    

cost_savings = 0
for i in range(CSIZE):

    if i%500 == 0:
        print(i)

    max_sc = score_list[-1]
    x,y,z  = score_map[max_sc][0]
    I[x][y][z] = 1
    score_list = score_list[:-1]
    score_map[max_sc] = [u for u in score_map[max_sc] if u != [x,y,z]]

    cost_savings += max_sc

    for n in neighbours:
        x1, y1, z1 = (x + n[0])%20, (y + n[1])%20, z
        A[x1][y1][z1] = 1
        recompute_scores(x1, y1, z1)
        for nn in neighbours:
            x2, y2, z2 = (x1 + nn[0])%20, (y1 + nn[1])%20, z
            recompute_scores(x2, y2, z2)

cost_save_per_req = float(cost_savings)/total_reqs

f = open(str(CSIZE) + ".txt", "w")
f.write(str(MISS - cost_save_per_req))
    
    
    



