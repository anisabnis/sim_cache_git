import matplotlib.pyplot as plt
import sys

prev_obj = 100000
objectives = []
xaxis = []
dir = sys.argv[1]

for i in range(1,193):
    f = open(dir + "/" + str(i * 2000) + ".txt", "r")
    l = f.readline().strip().split(" ")
    try :
        prev_obj = float(l[2])
        xaxis.append(i*2000)
        objectives.append(prev_obj)
    except:
        xaxis.append(i*2000)
        objectives.append(prev_obj)


print(xaxis)
print(objectives)
plt.plot(xaxis, objectives)
plt.ylabel("Objective")
plt.xlabel("Number of iterations")
plt.grid()
plt.savefig(dir + "/objective.png")
