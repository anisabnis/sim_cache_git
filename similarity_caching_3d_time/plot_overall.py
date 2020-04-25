import sys
import matplotlib.pyplot as plt

lru = []
#sgd = []
greedy = []
#sgd_001 = []
sgd_005 = []

sizes=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000]

for s in sizes:

#     f1 = open("20_0.1_cache_real_" + str(s) + "_dual_14_Warship/" + "objective.txt", "r")
#     i = 0
#     first = False
#     scores = []
#     for l in f1:
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
#     sgd.append(avg)


#     f4 = open("20_0.01_cache_real_" + str(s) + "_dual_14_Warship/" + "objective.txt", "r")
#     i = 0
#     first = False
#     scores = []
#     for l in f4:
#         i += 1
#         if i < 10:
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
#     avg = float(avg)/(i-10)
#     sgd_001.append(avg)


    f5 = open("20_0.05_cache_real_" + str(s) + "_dual_14_Warship/" + "objective.txt", "r")
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
    sgd_005.append(avg)


    i = 0
    f2 = open("20_0.01_cache_real_" + str(s) + "_dual_14_Warship_simple/objective.txt", "r")
    first = False
    scores = []
    for l in f2:
        i += 1
        if i < 50:
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
    avg = float(avg)/(i-50)
    lru.append(avg)


    f3 = open(str(s) + ".txt", "r")
    l = f3.readline()
    l = l.strip().split(" ")
    avg = float(l[0])
    greedy.append(avg)


#print(sgd)
#print(lru)
#print(greedy)

plt.plot(range(len(sizes)), lru, marker = "o", label="lru")
#plt.plot(range(len(sizes)), sgd, marker = "x", label="sgd")
#plt.plot(range(len(sizes)), sgd_001, marker = "1", label="sgd_0.01")
plt.plot(range(len(sizes)), sgd_005, marker = "2", label="sgd_0.05")
plt.plot(range(len(sizes)), greedy, marker = "^", label="greedy")
plt.xticks(range(len(sizes)), sizes)
plt.grid()
plt.legend()
plt.savefig("overall_score.png")
