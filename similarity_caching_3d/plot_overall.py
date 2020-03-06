import sys
import matplotlib.pyplot as plt

lru = []
sgd = []
greedy = []

sizes=[2000, 4000, 6000, 8000, 10000, 12000, 14000]

for s in sizes:

    f1 = open("20_0.1_cache_real_" + str(s) + "_dual_14_Warship/" + "objective.txt", "r")
    i = 0
    first = False
    scores = []
    for l in f1:
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
    sgd.append(avg)

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


    f3 = open("greedy_" + str(s) + ".txt", "r")
    l = f3.readline()
    l = l.strip().split(" ")
    avg = float(l[1])
    greedy.append(avg)


print(sgd)
print(lru)
print(greedy)

plt.plot(range(len(sizes)), lru, marker = "o", label="lru")
plt.plot(range(len(sizes)), sgd, marker = "x", label="sgd")
plt.plot(range(len(sizes)), greedy, marker = "^", label="greedy")
plt.xticks(range(len(sizes)), sizes)
plt.grid()
plt.legend()
plt.savefig("overall_score.png")
