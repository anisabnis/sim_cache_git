max_x=0
max_y=0
max_z=0
f = open("trace.txt", "r")
for l in f:
    l = l.strip().split(" ")
    z = int(l[1])
    x = int(l[3])
    y = int(l[4])

    if z > max_z:
        max_z = z
    if x > max_x:
        max_x = x
    if y > max_y:
        max_y = y

print(max_x, max_y, max_z)
