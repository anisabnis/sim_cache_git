f = open("2000_cache_contents.txt", "r")
s = set()
for l in f:
    l = l.strip().split(" ")
    s.add(l[0] + "." + l[1] + "." + l[2])
print(len(s))
