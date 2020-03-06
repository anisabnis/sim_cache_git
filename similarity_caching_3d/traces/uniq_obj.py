objects = set()
f = open("trace_20.txt", "r")
for l in f:
    l = l.strip().split(" ")
    req = l[1] + ":" + l[3] + ":" + l[4]
    objects.add(req)
print(len(objects))

