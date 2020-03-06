import sys
from collections import defaultdict
import bisect
import numpy as np
import matplotlib.pyplot as plt

cost_reduction = defaultdict(lambda : 0)
no_requests = defaultdict(lambda : 0)

exp_trace = sys.argv[1]
c_size = int(sys.argv[2])

f = open("new_traces/trace_" + str(exp_trace) + "_shuffle.txt", "r")

T = 8
A = 1

## grid
##
neighbour_saved = defaultdict(lambda : 0)
total_req_cost = 0
for l in f:
    total_req_cost += 8
    l = l.strip().split(" ")
    x, y, z = int(l[3]), int(l[4]), int(l[1])
    tile = str(x) + "." + str(y) + "." + str(z)
    cost_reduction[tile] += T
    no_requests[tile] += 1
    neighbour_saved[tile] = 0

    neighbours = [[-1,0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]

    for n in neighbours:
        if z + n[2] >= 500 or z + n[2] < 0:
            continue

        tile = str((x + n[0])%20) + "." + str((x + n[1])%20) + "." + str((z + n[2]))
        cost_reduction[tile] += (T-A)


print("number tiles : ", len(list(cost_reduction)))
## freq sorted list
## freq map
## Build the above two data structures
score_list = []
score_map = defaultdict(lambda : [])

for tile in cost_reduction:
    bisect.insort(score_list, cost_reduction[tile])
    score_map[cost_reduction[tile]].append(tile)


cache = []
zs = defaultdict(lambda : 0)

f1 = open(str(c_size) + "_cache_contents.txt", "w")
total_cost_saved = 0

for i in range(c_size):

    max_sc = score_list[-1]
    total_cost_saved += max_sc

    no_tiles = len(score_map[max_sc])
    r_no = np.random.randint(0, no_tiles)
    req_tile = score_map[max_sc][r_no]
    cache.append(req_tile)    

    r = req_tile.split(".")
    x,y,z = int(r[0]), int(r[1]), int(r[2])

    f1.write(str(x) + " " + str(y) + " " + str(z))
    f1.write("\n")
    zs[z] += 1

    ## Delete the tile you add in cache
    score_list = score_list[:-1]
    score_map[max_sc] = [u for u in score_map[max_sc] if u != req_tile]
    del cost_reduction[req_tile]

    n_x = no_requests[req_tile]
    ## Recompute the scores of the neighbouring tiles
#    neighbours = [[-1,0], [1, 0], [0, -1], [0, 1]]  
    neighbours = [[-1,0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]

    for n in neighbours :
        
        if z + n[2] >= 500 or z + n[2] < 0:
            continue
 
        n_tile = str((x + n[0])%20) + "." + str((y + n[1])%20) + "." + str(z + n[2])

        if n_tile in cost_reduction:

            n_y = no_requests[n_tile]

            n_score = cost_reduction[n_tile]

            del score_list[bisect.bisect_left(score_list, n_score)]
            score_map[n_score] = [u for u in score_map[n_score] if u != n_tile]            

            if len(score_map[n_score]) == 0:
                del score_map[n_score]

            n_score_update = n_score - ((n_x)*(T - A))
            if neighbour_saved[n_tile] == 0:
                n_score_update -= (n_y *(T-A))
                neighbour_saved[n_tile] = 0

            bisect.insort(score_list, n_score_update)
            score_map[n_score_update].append(n_tile)
            cost_reduction[n_tile] = n_score_update

f1.close()

z_keys = list(zs.keys())
z_keys.sort()
count = []
for z in z_keys:
    count.append(zs[z])
count = np.cumsum(count)
plt.plot(z_keys, count)
plt.xlabel("Segment Time")
plt.ylabel("Cumulative sum of number of segments cached for each time unit")
plt.grid()
plt.savefig("zs.png")

f.close()
f = open("new_traces/trace_" + str(exp_trace) + "_shuffle.txt", "r")

no_hits = 0
no_approx = 0
no_misses = 0

cost = 0
no_req = 0
for l in f:
    no_req += 1

    if no_req > 600000:
        break

    if no_req%10000 == 0:
        print(no_req)

    l = l.strip().split(" ")
    x, y, z = int(l[3]), int(l[4]), int(l[1])
    tile = str(x) + "." + str(y) + "." + str(z)

    if tile not in cache:
        neighbours = [[-1,0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
 
        #neighbours = [[-1,0], [1, 0], [0, -1], [0, 1]]  
        approx = False
        for n in neighbours:
            if z + n[2] >= 500 or z + n[2] < 0:
                continue
 
            n_tile = str((x + n[0])%20) + "." + str((y + n[1])%20) + "." + str(z + n[2])
            if n_tile in cache:
                cost += A
                approx = True
                no_approx += 1
                break

        if approx == False:
            no_misses += 1
            cost += T

    else:
        no_hits += 1

print(c_size, float(cost)/no_req, (float(total_cost_saved)*8)/total_req_cost)

o_file = "greedy_" + str(c_size) + ".txt"
f = open(o_file, "w")
f.write(str(c_size) + " " + str(float(cost)/no_req))
f.write("\n")
f.write(str(no_hits) + " " + str(no_approx) + " " + str(no_misses))
f.close()

# print("Cache contents : ")
# for c in cache:
#     print(c)

    






