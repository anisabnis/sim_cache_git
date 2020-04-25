import sys
import bisect
import numpy as np
import random

eye = sys.argv[1]
filename = sys.argv[2]

trace = []

for rep in range(40):

    f = open(eye + "/" + filename + "_fixations.csv", "r")
    f.readline()    
    
    def convert_to_tile(lat, long, x, y):
        return int(lat * 100)/x, int(long * 100)/x

    offset = 0
    user_data = []

    for l in f:
        l = l.strip().split(",")        
        u = int(l[0])

        if u == 0:
            ### Add to trace here
            start = np.random.randint(0, 5000)
            len_dat = len(user_data)
            if len(user_data) != 0:
                start_point = random.randint(0, len_dat)
                for i in range(len_dat):
                    j = (start_point + i)%len_dat
                    start += user_data[i][3]
                    req = [start, user_data[j][0], eye, user_data[j][1], user_data[j][2]]
                    bisect.insort(trace, req)                
                user_data = []

        lat = float(l[1])
        long = float(l[2])
        duration = float(l[4])
        f_st = int(l[5])
        f_end = int(l[6])
        
        time_inc = float(duration)/(f_end - f_st + 1)
    
        for i in list(range(f_st, f_end+1)):
            x, y = convert_to_tile(lat, long, 5, 5)
            bisect.insort(user_data, [i, x, y, time_inc])


#random.shuffle(trace)

f = open("trace_" + str(filename) + ".txt", "w")
for t in trace:
    f.write(" ".join([str(x) for x in t]))
    f.write("\n")
f.close()



    
