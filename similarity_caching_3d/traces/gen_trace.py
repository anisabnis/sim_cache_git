import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import bisect
import random
import sys

dat=sio.loadmat("FULLdata_per_video_frame.mat")
min_lat = 20000
max_lat = -20000
min_long = 20000
max_long = -20000
segments = []


for vid in dat:
    req_dat = dat[vid]

    lat_var = []
    long_var = []

    max_len = 0

    max_len = max(max_len, len(req_dat))

    segments.append(len(req_dat))

    for hm in req_dat:

        lats = []
        longs = []

        for i in range(len(hm)/2):
            lats.append(hm[2*i])
            longs.append(hm[2*i + 1])

            if hm[2*i] < min_lat:
                min_lat = hm[2*i]

                
            if hm[2*i + 1] < min_long:
                min_long = hm[2*i +1]

        lat_var.append(np.var(lats))
        long_var.append(np.var(longs))




print(min_lat, max_lat, min_long, max_long)

segments.sort()
plt.plot(segments)
plt.xlabel("Video ID")
plt.ylabel("Number of segments")
plt.grid()
plt.savefig("DurationDistribution.png")


def get_frame_id(x, y):
    x = int(x/10) + 9
    y = int(y/10) + 18
    return x,y


users = range(464)
videos = list(dat.keys())
videos = ["StarWars2"]
random.shuffle(videos)

for v1 in videos:

    trace = []


    for u in users:
        u = u % 57
        req_dat = dat[v1]
        timestamp = random.randint(0, len(req_dat)/4)
        random.shuffle(videos)        
        i = 0            

        try:
            for hm in req_dat:                                    
                nt = random.random()/2
                timestamp += nt
                x, y = get_frame_id(hm[2*u], hm[2*u + 1])
                                    
                if x < -90 or x > 90:
                    continue
                
                if y < -180 or y > 180:
                    continue
                                
                req = [timestamp, i, v1, x, y]
                bisect.insort(trace, req)
                i += 1
                
        except:
            pass



    print(v1)
    f = open("trace_" + v1 + ".txt", "w")
    for t in trace:
        f.write(" ".join([str(x) for x in t]))
        f.write("\n")
    f.close()
    
#     plt.plot(lat_var, label="Latitude")
#     plt.plot(long_var, label="Longitude")
#     plt.xlabel("time")
#     plt.ylabel("variance")
#     plt.legend()
#     plt.savefig(vid + ".png")
#     plt.clf()
    
