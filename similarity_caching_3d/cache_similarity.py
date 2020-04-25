from collections import defaultdict
import matplotlib.pyplot as plt

csize=[4000, 10000, 12000]
for c in csize:

    ## First read the contents greedy added in cache    
    greedy = defaultdict(lambda : 1)
    f = open(str(c) + "_content.txt", "r")
    for l in f:
        l = l.strip().split(" ")
        x, y, z = int(l[0]), int(l[1]), int(l[2])
        coord = str(x) + "." + str(y) + "." + str(z)
        greedy[coord] = 1
    f.close()
        

    ## SGD
    score_sgd = []
    for i in range(5,300):
        f = open("20_0.05_cache_real_" + str(c) + "_lru_14_Warship_0.1/cache_contents_" + str(i) + ".txt", "r")
        score = 0
        for l in f:
            l = l.strip().split(" ")
            x, y, z = int(l[0]), int(l[1]), int(l[2])
            coord = str(x) + "." + str(y) + "." + str(z) 
            if coord in greedy:
                score += 1

        score = float(score)/c
        score_sgd.append(score)

    ## LRU
    score_lru = []
    for i in range(5,300):
        f = open("20_0.01_cache_real_" + str(c) + "_lru_14_Warship_simple/cache_contents_" + str(i) + ".txt", "r")
        score = 0
        for l in f:
            l = l.strip().split(" ")
            x, y, z = int(l[0]), int(l[1]), int(l[2])
            coord = str(x) + "." + str(y) + "." + str(z) 
            if coord in greedy:
                score += 1

        score = float(score)/c
        score_lru.append(score)
        

    plt.plot(score_sgd, label="SGD")
    plt.plot(score_lru, label="LRU")
    plt.legend()
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Cache Similarity")
    plt.savefig(str(c) + "_cache_sim.png")
    plt.clf()
