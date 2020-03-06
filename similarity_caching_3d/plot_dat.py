import matplotlib.pyplot as plt
f = open("data", "r")
xs = []
ys =[]
for l in f:
    l = l.strip().split(",")
    xs.append(int(l[0]))
    ys.append(float(l[1]))

plt.plot(xs, ys)
plt.xlabel("Cache size")
plt.ylabel("Average cost incurred")
plt.grid()
plt.savefig("CostVsCSize.png")
              
