import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

dat=sio.loadmat("FULLdata_per_video_frame.mat")

for vid in dat:
    req_dat = dat[vid]

    lat_var = []
    long_var = []

    for hm in req_dat:

        lats = []
        longs = []

        for i in range(len(hm)/2):
            lats.append(hm[2*i])
            longs.append(hm[2*i + 1])

        lat_var.append(np.var(lats))
        long_var.append(np.var(longs))


    plt.plot(lat_var, label="Latitude")
    plt.plot(long_var, label="Longitude")
    plt.xlabel("time")
    plt.ylabel("variance")
    plt.legend()
    plt.savefig(vid + ".png")
    plt.clf()
    
