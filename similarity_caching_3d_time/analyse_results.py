
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import BoundaryNorm

for d in ["20_0.05_cache_real_2000_dual_14_Warship_0.1", "20_0.05_cache_real_4000_dual_14_Warship_0.1", "20_0.05_cache_real_6000_dual_14_Warship_0.1", "20_0.05_cache_real_8000_dual_14_Warship_0.1", "20_0.05_cache_real_10000_dual_14_Warship_0.1", "20_0.05_cache_real_12000_dual_14_Warship_0.1"]:

    keyword = d.split("_")[4]
    for ite in [300, 400, 450]:
        xs = defaultdict(list)
        ys = defaultdict(list)
        
        max_z = 0
        max_x = 20
        max_y = 20
        
        f = open(d + "/cache_contents_" + str(ite) + ".txt", "r")
        for l in f:
            l = l.strip().split(" ")
            z = int(float(l[2]))
            
            if z > max_z:
                max_z = z
            
            x = int(float(l[0]))
            y = int(float(l[1]))
            
            xs[z].append(x)
            ys[z].append(y)

        arr_z = range(max_z)
        arr_x = range(max_x)
        arr_y = range(max_y)

        intensity = np.zeros((max_z, max_x))
        for z in arr_z:
            if z in xs:
                if type(xs[z]) is int:
                    intensity[z][xs[z]] += 1
                else:
                    for x in xs[z]:
                        print(x)
                        intensity[z][x] += 1

        intensity = np.array(intensity)
        arr_z = np.array(arr_z)
        arr_x = np.array(arr_x)
        arr_y = np.array(arr_y)
        cmap = plt.get_cmap('hsv')
        plt.pcolormesh(arr_x, arr_z, intensity, cmap=cmap)
        plt.savefig(str(ite) + "_" +  keyword + "_Xs.png")
        
        
        intensity = np.zeros((max_z, max_y ))
        for z in arr_z:
            if z in ys:
                if type(ys[z]) is int:
                    intensity[z][ys[z]] += 1
                else:
                    for y in ys[z]:
                        intensity[z][y] += 1

        intensity = np.array(intensity)
        cmap = plt.get_cmap('hsv')
    # Construct 2D histogram from data using the 'plasma' colormap
        plt.pcolormesh(arr_y, arr_z, intensity, cmap=cmap)
        plt.savefig(str(ite) + "_" + keyword + "_Ys.png")
