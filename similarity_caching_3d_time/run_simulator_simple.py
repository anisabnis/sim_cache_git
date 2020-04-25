import argparse
import copy
import os
import time
import sys
from obj_catalogue import *
from gradient_descent import *
import operator

experiment_type = sys.argv[1]
policy = sys.argv[3]
reset_interval = int(sys.argv[2])
file_name_extension = sys.argv[4]
capacity = int(file_name_extension)
v_id = sys.argv[5]

if policy == "dual" or policy == "fifo" or policy == "lru":
    from cacheGrid3d import *
else:
    from cacheReal import *


class Simulator:
    def __init__(self, dim, capa, no_objects, alpha, iter, update_interval, learning_rate):

        self.grid_x = 20

        self.grid_y = 20

        self.grid_z = 502

        self.obj_catalogue = ObjectCatalogueGrid3d(self.grid_x, self.grid_y, self.grid_z, experiment_type, v_id)        

        self.cache = CacheGridReal(capacity, dim, learning_rate, True, [self.grid_x, self.grid_y, self.grid_z], policy)

        self.cache.initializeIterativeSearch([self.grid_x, self.grid_y, self.grid_z])
 
        self.iter = iter

        self.u_interval = update_interval

        self.descent = StochasticGradientDescent(learning_rate, [self.grid_x, self.grid_y, self.grid_z])

        self.learning_rate = learning_rate

        os.system("mkdir " + str(self.grid_x) + "_" + str(learning_rate) + "_" + experiment_type + "_" + file_name_extension + "_" + policy + "_" + v_id + "_simple")        
        
        self.cache_capacity = capacity


    def write_stat(self, f,  i, objective_value, content_cache, cache_hits, approximated, cache_misses):
        f.write(str(i) + " " + str(objective_value) + " " + str(content_cache) + " " + str(cache_hits) + " " + str(approximated) + " " + str(cache_misses))
        f.write("\n")
        f.flush()

    def simulate(self):

        def l1_dist(p1, p2):
            return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]), ord=1) + 1 * abs(p1[2] - p2[2]) 

        objective = [] 
        objective_value = 0

        epsilon = 0.01
        
        count = 0
        prev_i = 0

        jump_interval = 2000

        f = open(str(self.grid_x) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '_' + policy + "_" + v_id + "_simple" +'/' + str("objective") + '.txt', 'w')                
        cost = 0

        Threshold = 1 ## Decide a value
        Threshold_N = 8

        cache_initialized = False

        cache_hits = 0
        cache_misses = 0
        approximated = 0
                     
        seq = 0
        for i in range(1, self.iter):

            if i - prev_i >= jump_interval:

                if cache_initialized == False:
                    objective_value1 = 0
                else :
                    objective_value1 = 0
                    
                    objective_value = float(cost)/(jump_interval)
            
                    if i < 100000 and i == 10 * jump_interval:
                        jump_interval *= 10
                    elif i == 100000 and i == 10 * jump_interval:
                        pass
                    elif i == 100 * jump_interval:
                        jump_interval *= 10


                    curr = self.cache.getCurrent()

                    f2 = open(str(self.grid_x) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '_' + policy + "_" + v_id + '_' + "simple" +'/' + str("cache_contents_") + str(seq) + '.txt', 'w')                

                    seq += 1
                    content_cache = self.cache.obj_pos.printCacheContents(policy, curr, f2)

                    f2.close()

                    self.write_stat(f, i, objective_value, content_cache, cache_hits, approximated, cache_misses)

                    cost = 0

                    prev_i = i

                    print("iter : ", i, "objective : ", objective_value)


            objs = self.obj_catalogue.getRequest()                    

            for obj in objs:
                obj = obj[:-1]

                if self.cache.getObjCount() < self.cache_capacity and cache_initialized == False:
                    self.cache.insertInit(obj, i, policy)
                    continue
                else:
                    cache_initialized = True
                
                if obj[-1] == "n":
                    ## Can only serve with the actual object
                    pass
                else:
                    pos = obj

                    ## Find the nearest object in real cache
                    [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearestReal(pos) 


                    if nearest_obj != "Not found":
                        self.cache.updateVirtualObjectAndFreq(nearest_obj, nearest_obj, mapped_x, mapped_y, i, policy, round(float(dst), 3), reset_interval, pos, Threshold)

                        dst3 = l1_dist(nearest_obj, pos)
                        if dst3 <= Threshold:
                            cost += dst3
                            if dst3 == 0:
                                cache_hits += 1
                            else:
                                approximated += 1
                        else:
                            cost += Threshold_N
                            self.cache.insert(obj, i, policy)                          
                            cache_misses += 1
                    else:
                        cost += Threshold_N
                        self.cache.insert(obj, i, policy)
                        cache_misses += 1

                    
                                              
        
s = Simulator(2, capacity, 100, 0.4, 100000000, 1, 0.01)
s.simulate()                



                



        
