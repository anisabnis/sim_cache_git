import sys
import numpy as np

grid_x = int(sys.argv[1])
grid_y = int(sys.argv[2])

f = open("cache_pos.txt", "w")
for i in range(1000):
    ii = np.random.randint(0, grid_x)
    jj = np.random.randint(0, grid_y)
    f.write(str(ii) + " " + str(jj) + "\n")
f.close()
    
