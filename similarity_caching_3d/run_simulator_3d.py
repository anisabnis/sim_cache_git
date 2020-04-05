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
jump = float(sys.argv[6])

if policy == "dual" or policy == "fifo" or policy == "lru":
    from cacheGrid3d import *
else:
    from cacheReal import *


class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):

        self.grid_x = 20

        self.grid_y = 20

        self.grid_z = 500

        self.obj_catalogue = ObjectCatalogueGrid3d(self.grid_x, self.grid_y, self.grid_z, experiment_type, v_id)        

        self.cache = CacheGridReal(capacity, dim, learning_rate, True, [self.grid_x, self.grid_y, self.grid_z], policy)

        self.cache.initializeIterativeSearch([self.grid_x, self.grid_y, self.grid_z])
 
        self.iter = iter

        self.u_interval = update_interval

        self.descent = StochasticGradientDescent(learning_rate, [self.grid_x, self.grid_y, self.grid_z])

        self.learning_rate = learning_rate

        os.system("mkdir " + str(self.grid_x) + "_" + str(learning_rate) + "_" + experiment_type + "_" + file_name_extension + "_" + policy + "_" + v_id + "_" + str(jump))                
        self.cache_capacity = capacity


    def write_stat(self, f,  i, objective_value, content_cache, evictions, cache_misses, usefulness, cache_hits, approximated, no_fetches):
        f.write(str(i) + " " + str(objective_value) + " " + str(content_cache) + " " + str(evictions) + " " + str(cache_misses) + " " + str(usefulness) + " " + str(cache_hits) + " " + str(approximated) + " " + str(no_fetches))
        f.write("\n")
        f.flush()

    def simulate(self):

        def l1_dist(p1, p2):
            return np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2]), ord=1) + 1 * abs(p1[2] - p2[2]) 

        objective = [] 
        objective_value = 0

        epsilon = jump
        
        count = 0
        prev_i = 0

        jump_interval = 2000

        f = open(str(self.grid_x) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '_' + policy + "_" + v_id + "_" + str(jump) + '/' + str("objective") + '.txt', 'w')                
        cost = 0
        evictions = 0
        cache_misses = 0
        approximated = 0
        cache_hits = 0

        Threshold = 1 ## Decide a value
        Threshold_N = 8

        no_fetches = 0

        cache_initialized = False
                     
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

                    f2 = open(str(self.grid_x) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '_' + policy + "_" + v_id + "_" + str(jump) + '/' + str("cache_contents_") + str(seq) + '.txt', 'w')                
                    content_cache = self.cache.obj_pos.printCacheContents(policy, curr, f2)
                    f2.close()
                    seq += 1
                    usefulness = self.cache.getUsefulness(policy, curr)

                    self.write_stat(f, i, objective_value, content_cache, evictions, cache_misses, usefulness, cache_hits, approximated, no_fetches)

                    cost = 0
#                    cache_misses = 0
#                    evictions = 0

                    prev_i = i

                    print("iter : ", i, "objective : ", objective_value, "usefulness : ", usefulness)


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
                    
                    updated_real_obj = False
                    [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearestVirtual(pos)                                        

                    #print("Nearest virtual object : ", nearest_obj, obj)
                    if nearest_obj != "Not found":

                        new_object_loc = self.descent.descent(nearest_obj, obj, "gaussian_real", policy)                
                        
                        self.cache.updateVirtualObjectAndFreq(nearest_obj, new_object_loc, mapped_x, mapped_y, i, policy, round(dst,2), reset_interval, pos, Threshold)                                                   
                        ## Virtual Hit
                        if dst <= Threshold: 
                    
                            mapped_real_object = self.cache.getReal(new_object_loc)            
                        
                            ## Find the distance between the requested object and the object in cache which could serve the request
                            dist = l1_dist(pos, mapped_real_object)

                            ## Find the distance between the mapped real object to the virtual object
                            dst2 = l1_dist(np.array([mapped_real_object[0], mapped_real_object[1], mapped_real_object[2]]), np.array([nearest_obj[0], nearest_obj[1], nearest_obj[2]]))

                            ## The requested object is a better candidate to save in the cache as compared to what is 
                            ## already present
                            if dist < dst2:
                                check_if_in_real_cache = (obj[0] == mapped_real_object[0] and obj[1] == mapped_real_object[1] and obj[2] == mapped_real_object[2])
                                z = np.random.random()                    
                                if check_if_in_real_cache == False:# and z < 0.1:
                                    cost += Threshold_N
                                    new_real_obj = obj
                                    virtual_obj = new_object_loc                            
                                    orig_real_obj = mapped_real_object                                    
                                    updated_real_obj = True
                                    self.cache.updateRealObject(new_real_obj, i, policy, virtual_obj, orig_real_obj)
                                    no_fetches += 1


                        ## Find the nearest object in real cache
                    [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearestReal(pos) 
                        #print("Nearest real object : ", nearest_obj, obj)
                    z = np.random.random()                    
                    if nearest_obj != "Not found":
                        dst3 = l1_dist(nearest_obj, pos)

                        if dst3 <= Threshold:
                            if dst3 == 0:
                                cache_hits += 1
                            else:
                                approximated += 1

                                if updated_real_obj == False:
                                    cost += dst3
                        else:
                            cache_misses += 1
                            if z <= epsilon and updated_real_obj == False:
                                self.cache.insert(obj, i, policy)
                                nearest_object = pos
                                dst3 = 0
                                cost += Threshold_N
                                evictions += 1
                                updated_real_obj = True
                            if updated_real_obj == False:
                                cost += Threshold_N

                    elif z <= epsilon and updated_real_obj == False:
                    #elif z <= epsilon * min(float(1)/Threshold, 1) and updated_real_obj == False:
                        self.cache.insert(obj, i, policy)
                        nearest_object = pos
                        dst3 = 0
                        if updated_real_obj == False:
                            cost += Threshold_N
                        evictions += 1
                
                    else:
                        cache_misses += 1
                        if updated_real_obj == False:
                            cost += Threshold_N
                            
        
s = Simulator(2, capacity, 100, 0.1, 100000000, 1, 0.0)
s.simulate()                



                



        
