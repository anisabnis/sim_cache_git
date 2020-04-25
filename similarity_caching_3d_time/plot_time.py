import matplotlib.pyplot as plt

csize=[4000, 10000]
#csize=[10000]
for c in csize:
    sgd = []
    f1 = open("20_0.05_cache_real_" + str(c) + "_dual_14_Warship_0.1/objective.txt", "r")
    for l in f1:
        l = l.strip().split(" ")
        obj = float(l[1])
        sgd.append(obj)

    lru = []
    f2 = open("20_0.01_cache_real_" + str(c) + "_lru_14_Warship_simple/objective.txt", "r")
    for l in f2:
        l = l.strip().split(" ")
        obj = float(l[1])
        lru.append(obj)

    plt.plot(lru, label="lru")
    plt.plot(sgd, label="sgd")
    plt.xlabel("Time")
    plt.ylabel("Objective")
    plt.grid()
    plt.legend()
    plt.savefig(str(c) + "_time.png")
    plt.clf()
