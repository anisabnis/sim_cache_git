import sys
from collections import defaultdict
import bisect
import numpy as np
import matplotlib.pyplot as plt
import random

cost_reduction = defaultdict(lambda : 0)
no_requests = defaultdict(lambda : 0)

exp_trace = sys.argv[1]
c_size = int(sys.argv[2])

f = open("new_traces/trace_" + str(exp_trace) + "_shuffle1_norepeat.txt", "r")

T = 8
A = 1

## grid
##
neighbour_saved = defaultdict(lambda : 0)
total_req_cost = 0
total_reqs = 0

intensity = np.zeros((20, 20))

for l in f:

    total_req_cost += 8
    l = l.strip().split(" ")
    x, y, z = int(l[3]), int(l[4]), int(l[1])
    intensity[x][y] += 1
    tile = str(x) + "." + str(y) + "." + str(z)
    cost_reduction[tile] += T
    no_requests[tile] += 1
    neighbour_saved[tile] = 0
    total_reqs += 1

    
#    if total_reqs > 100000:
#        break

    neighbours = [[-1,0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
    for n in neighbours:
        if z + n[2] >= 502 or z + n[2] < 0:
            continue

        tile = str((x + n[0])%20) + "." + str((y + n[1])%20) + "." + str((z + n[2]))
        cost_reduction[tile] += (T-A)


intensity = np.array(intensity)
cmap = plt.get_cmap('hsv')
plt.pcolormesh(np.array(range(20)), np.array(range(20)), intensity, cmap=cmap)
plt.savefig("cache_content.png")

sum_requests= 0
for t in no_requests:
    sum_requests += no_requests[t]
print(sum_requests)

print("number tiles : ", len(list(cost_reduction)))

#print(cost_reduction)

## freq sorted list
## freq map
## Build the above two data structures
score_list = []
score_map = defaultdict(lambda : [])

for tile in cost_reduction:
    #print(tile, cost_reduction[tile])
    bisect.insort(score_list, cost_reduction[tile])
    score_map[cost_reduction[tile]].append(tile)

#print(score_list)
#print(score_map)

cache = []
zs = defaultdict(lambda : 0)
f1 = open(str(c_size) + "_cache_contents.txt", "w") 
total_cost_saved = 0

print("---------------")

for i in range(c_size):

    max_sc = score_list[-1]
    total_cost_saved += max_sc

    no_tiles = len(score_map[max_sc]) - 1
    #print(no_tiles, max_sc, score_map[max_sc])
    
    r_no = random.randint(0, no_tiles)
    req_tile = score_map[max_sc][0]
    cache.append(req_tile)    

    #print("Adding to cache : ", req_tile, max_sc)

    r = req_tile.split(".")
    x,y,z = int(r[0]), int(r[1]), int(r[2])
    f1.write(str(x) + " " + str(y) + " " + str(z) + " " + str(max_sc))
    f1.write("\n")
    zs[z] += 1

    ## Delete the tile you add in cache
    score_list = score_list[:-1]
    score_map[max_sc] = [u for u in score_map[max_sc] if u != req_tile]
    if len(score_map[max_sc]) == 0:
        del score_map[max_sc]
    del cost_reduction[req_tile]

    n_x = no_requests[req_tile]
    #Recompute the scores of the neighbouring tiles
    neighbours = [[-1,0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]

    
    for n in neighbours :
        
        if z + n[2] >= 502 or z + n[2] < 0:
            continue

        n_tile = str((x + n[0])%20) + "." + str((y + n[1])%20) + "." + str(z + n[2])
        
        if n_tile in cost_reduction:

            n_y = no_requests[n_tile]

            n_score = cost_reduction[n_tile]
            
            del score_list[bisect.bisect_left(score_list, n_score)]
            score_map[n_score] = [u for u in score_map[n_score] if u != n_tile]            
            if len(score_map[n_score]) == 0:
                del score_map[n_score]
            n_score_update = n_score - ((n_x + n_y)*(T - A))
            bisect.insort(score_list, n_score_update)
            score_map[n_score_update].append(n_tile)
            cost_reduction[n_tile] = n_score_update


        x1, y1, z1 = [int(m) for m in n_tile.split(".")]
        
        for n2 in neighbours :

            if z1 + n2[2] >= 502 or z1 + n2[2] < 0:
                continue

            n2_tile = str((x1 + n2[0])%20) + "." + str((y1 + n2[1])%20) + "." + str(z1 + n2[2])

            if n2_tile in cost_reduction:

                curr_score = cost_reduction[n2_tile]                    
                del score_list[bisect.bisect_left(score_list, curr_score)]
                score_map[curr_score] = [u for u in score_map[curr_score] if u != n2_tile]            

                if len(score_map[curr_score]) == 0:
                    del score_map[curr_score]

                new_score = cost_reduction[n2_tile] - (T-A)*n_y
                cost_reduction[n2_tile] = new_score

                #print("2. Reassigning score of : ", n2_tile, curr_score, new_score)

                bisect.insort(score_list, new_score)
                score_map[new_score].append(n2_tile)                    
                    
f1.close()


print("objective : ", float(T * total_reqs - total_cost_saved)/total_reqs)

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
f = open("new_traces/trace_" + str(exp_trace) + "_shuffle1.txt", "r")

no_hits = 0
no_approx = 0
no_misses = 0

cost = 0
no_req = 0


print("Cache length : ", len(set(cache)))

# for l in f:
#     no_req += 1
    
# #    if no_req > 100000:
# #        break

#     if no_req%10000 == 0:
#         print(no_req)

#     l = l.strip().split(" ")
#     x, y, z = int(l[3]), int(l[4]), int(l[1])
#     tile = str(x) + "." + str(y) + "." + str(z)

#     if tile not in cache:
#         neighbours = [[-1,0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]]
 
#         #neighbours = [[-1,0], [1, 0], [0, -1], [0, 1]]  
#         approx = False
#         for n in neighbours:
#             if z + n[2] >= 500 or z + n[2] < 0:
#                 continue
 
#             n_tile = str((x + n[0])%20) + "." + str((y + n[1])%20) + "." + str(z + n[2])
#             if n_tile in cache:
#                 cost += A
#                 approx = True
#                 no_approx += 1
#                 break

#         if approx == False:
#             no_misses += 1
#             cost += T

#     else:
#         no_hits += 1

# print(c_size, float(cost)/no_req, (float(total_cost_saved)*8)/total_req_cost)

# o_file = "greedy_" + str(c_size) + ".txt"
# f = open(o_file, "w")
# f.write(str(c_size) + " " + str(float(cost)/no_req))
# f.write("\n")
# f.write(str(no_hits) + " " + str(no_approx) + " " + str(no_misses))
# f.close()

# print("Cache contents : ")
# for c in cache:
#     print(c)

    






