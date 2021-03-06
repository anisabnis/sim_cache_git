import sys
import bisect
import numpy as np
import random

eye = sys.argv[1]
filename1 = sys.argv[2]
filename2 = sys.argv[3]
trace = []

for rep in range(40):

    f = open(eye + "/" + filename1 + "_fixations.csv", "r")
    f.readline()    
    
    def convert_to_tile(lat, long, x, y):
        return int(lat * 100)/x, int(long * 100)/x

    offset = 0

    for l in f:
        l = l.strip().split(",")        
        u = int(l[0])

        if u == 0:
            offset = np.random.randint(0, 5000)

        lat = float(l[1])
        long = float(l[2])
        start = float(l[3]) + offset
        duration = float(l[4])
        f_st = int(l[5])
        f_end = int(l[6])
        
        time_inc = float(duration)/(f_end - f_st + 1)
    
        for i in list(range(f_st, f_end+1)):
            start = start + time_inc
            x, y = convert_to_tile(lat, long, 5, 5)
            req = [start, i, eye, x, y]
            bisect.insort(trace, req)



for rep in range(40):

    f = open(eye + "/" + filename2 + "_fixations.csv", "r")
    f.readline()    
    
    def convert_to_tile(lat, long, x, y):
        return (int(lat * 100)/x) + 100, int(long * 100)/x

    offset = 0

    for l in f:
        l = l.strip().split(",")        
        u = int(l[0])

        if u == 0:
            offset = np.random.randint(2000, 7000)

        lat = float(l[1])
        long = float(l[2])
        start = float(l[3]) + offset
        duration = float(l[4])
        f_st = int(l[5])
        f_end = int(l[6])
        
        time_inc = float(duration)/(f_end - f_st + 1)
    
        for i in list(range(f_st, f_end+1)):
            start = start + time_inc
            x, y = convert_to_tile(lat, long, 5, 5)
            req = [start, i, eye, x, y]
            bisect.insort(trace, req)


random.shuffle(trace)
f = open("trace_" + str(filename1) + "_" + str(filename2) + "_shuffle_cyclic_mix.txt", "w")
#for i in range(30):
for t in trace:
    f.write(" ".join([str(x) for x in t]))
    f.write("\n")
f.close()



    
