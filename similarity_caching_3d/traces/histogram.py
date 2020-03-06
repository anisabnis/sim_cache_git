import sys
video = sys.argv[1]
f = open("trace_" + str(video) + ".txt", "r")
from collections import defaultdict

lat_d = defaultdict(lambda : 0)
long_d = defaultdict(lambda : 0)

for l in f:
    l = l.strip().split(" ")
    lat = int(l[3])
    long = int(l[4])

    lat_d[lat] += 1
    long_d[long] += 1

lats = list(lat_d.keys())
lats.sort()
vals = []
for l in lats:
    vals.append(lat_d[l])    
print(lats)
print(vals)


longs = list(long_d.keys())
longs.sort()
vals = []
for l in longs:
    vals.append(long_d[l])    
print(longs)
print(vals)
