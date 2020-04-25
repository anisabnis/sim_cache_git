import sys
import matplotlib.pyplot as plt
from collections import defaultdict

lru = []
lru_simple = []
#sgd = []
greedy = []
#sgd_001 = []
#sgd_005 = []
sgd = defaultdict(lambda : [])
lru_dual = []
sizes=[2000, 4000, 6000, 8000, 10000, 12000]

eps = 0.1
for s in sizes:
    #for lr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.5]:
    for lr in [0.01, 0.02, 0.001, 0.05]:
    #for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]:
    #for eps in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print("20_" + str(lr) + "_cache_real_" + str(s) + "_dual_14_Warship_" + str(eps))
        f5 = open("20_" + str(lr) + "_cache_real_" + str(s) + "_dual_14_Warship_" + str(eps) + "/objective.txt", "r")
        i = 0
        first = False
        scores = []
        for l in f5:
            i += 1
            if i < 10:
                continue

            l = l.strip().split(" ")

            if first == False:
                first_req = int(l[0])
                first = True

            req_no = int(l[0])
            score = float(l[1])
            scores.append(score)

        total_req = req_no - first_req
        avg = sum(scores)
        avg = float(avg)/(i-10)
        sgd[lr].append(avg)

        
    i = 0
    print("20_0.01_cache_real_" + str(s) + "_lru_14_Warship_simple")
    f2 = open("20_0.01_cache_real_" + str(s) + "_lru_14_Warship_simple/objective.txt", "r")
    first = False
    scores = []
    for l in f2:
        i += 1
        if i < 10:
            continue

        l = l.strip().split(" ")

        if first == False:
            first_req = int(l[0])
            first = True

        req_no = int(l[0])
        score = float(l[1])
        scores.append(score)

    total_req = req_no - first_req
    avg = sum(scores)
    avg = float(avg)/(i-10)
    lru.append(avg)

#     i = 0
#     print("20_0.01_cache_real_" + str(s) + "_lru_14_Warship_simple")
#     f6 = open("20_0.01_cache_real_" + str(s) + "_dual_14_Warship_simple/objective.txt", "r")
#     first = False
#     scores = []
#     for l in f6:
#         i += 1
#         if i < 50:
#             continue

#         l = l.strip().split(" ")

#         if first == False:
#             first_req = int(l[0])
#             first = True

#         req_no = int(l[0])
#         score = float(l[1])
#         scores.append(score)

#     total_req = req_no - first_req
#     avg = sum(scores)
#     avg = float(avg)/(i-50)
#     lru_dual.append(avg)



#     i = 0
#     print("20_0.01_cache_real_" + str(s) + "_lru_14_Warship_simple_no_repeat")
#     f3 = open("20_0.01_cache_real_" + str(s) + "_lru_14_Warship_simple_no_update/objective.txt", "r")
#     first = False
#     scores = []
#     for l in f3:
#         i += 1
#         if i < 50:
#             continue

#         l = l.strip().split(" ")

#         if first == False:
#             first_req = int(l[0])
#             first = True

#         req_no = int(l[0])
#         score = float(l[1])
#         scores.append(score)

#     total_req = req_no - first_req
#     avg = sum(scores)
#     avg = float(avg)/(i-50)
#     lru_simple.append(avg)


    f4 = open(str(s) + ".txt", "r")
    l = f4.readline()
    l = l.strip().split(" ")
    avg = float(l[0])
    greedy.append(avg)


#print(sgd)
#print(lru)
#print(greedy)

plt.plot(range(len(sizes)), lru, marker = "o", label="lru")
#plt.plot(range(len(sizes)), lru_dual, marker="x", label="LeastScore")
#plt.plot(range(len(sizes)), sgd, marker = "x", label="sgd")
#plt.plot(range(len(sizes)), sgd_001, marker = "1", label="sgd_0.01")

# plt.plot(range(len(sizes)), lru_simple, marker="x", label="no_update")
for eps in sgd:
    plt.plot(range(len(sizes)), sgd[eps], label="sgd_"+str(eps))

plt.plot(range(len(sizes)), greedy, marker = "^", label="greedy")
plt.xticks(range(len(sizes)), sizes)
plt.grid()
plt.legend()
plt.savefig("overall_score.png")
