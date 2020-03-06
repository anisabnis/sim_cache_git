from cacheReal import *
import argparse
import copy
import os
import time
import sys

experiment_type = sys.argv[1]
file_name_extension = sys.argv[2]
policy = sys.argv[3]

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
        self.grid_d = 313

        self.obj_catalogue = ObjectCatalogueGrid(self.grid_d, self.grid_d, experiment_type)        

        self.cache = CacheGridReal(capacity, dim, learning_rate, True, [self.grid_d, self.grid_d])

        self.cache.initializeIterativeSearch([self.grid_d, self.grid_d])
 
        self.iter = iter

        self.u_interval = update_interval

        self.descent = StochasticGradientDescent(learning_rate, self.grid_d)

        self.plot =  Plots(experiment_type, file_name_extension)

        self.initial_points = self.cache.getAllPoints()    

        self.learning_rate = learning_rate

        os.system("mkdir " + str(self.grid_d) + "_" + str(learning_rate) + "_" + experiment_type + "_" + file_name_extension)        
        
        self.cache_capacity = capacity

    def write_stat(self, i, obj, f, cache_size, obj_val1, g_a, g_b):
        f.write(str(i) + "\t" + str(obj) + "\t" + str(obj_val1) + "\t" + str(cache_size) + "\t" + str(g_a) + "\t" + str(g_b))
        f.write("\n")
        f.flush()


    def write_stat_2(self, phy_hit, phy_miss, v_hit, v_miss, i_new, r_update, f):
        f.write(str(phy_hit) + " " + str(phy_miss) + " " + str(v_hit) + " " + str(v_miss) + " " + str(i_new) + " " + str(r_update) + "\n")
        f.flush()
                
    def write_rare_requests(self, req, np, f):
        f.write(' '.join([str(r) for r in req]))
        f.write(' ')
        f.write(' '.join([str(p) for p in np]))
        f.write('\n')
        f.flush()
        
    def write_distance_count(self, distance_count, f):
        for d in distance_count:
            f.write(str(d) + " " + str(distance_count[d]) + "\n")
            f.flush()

    def write_stat_debug(self, f2, obj, nearest_obj, mapped_x, mapped_y):
        f2.write("Request : " + ' '.join([str(x) for x in obj]) + " Nearest : " +  ' '.join([str(x) for x in nearest_obj]) + " Mapped : " +  str(mapped_x) + " " + str(mapped_y) + '\n')
        f2.flush()

    def simulate(self):

        def l1_dist(p1, p2):
            return np.linalg.norm(p1 - p2, ord=1)

        objective = [] 
        objective_value = 0
        
        count = 0
        prev_i = 0
        jump_interval = 100

        number_obj = len(self.cache.getAllPoints())

        f = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '/' + str("objective") + '.txt', 'w')                
        f2 = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '/' + str("debug") + '.txt', 'w')                   
        f3 = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '/' + str("req_stat") + '.txt', 'w')                            
        cost = 0

        Threshold = 10

        cache_initialized = False

        virtual_hit = 0
        virtual_miss = 0
        physical_hit = 0
        physical_miss = 0
        insert_new = 0
        real_update = 0
                     
        for i in range(1,self.iter):

            if i % self.u_interval == 0:                                

                if i - prev_i >= jump_interval:
                    
                    if cache_initialized == False:
                        objective_value1 = 0
                    else :
                        #objective_value = self.obj_catalogue.objective_l1_iterative_threaded(self.cache, experiment_type)                
                        #objective_value1 = objective_value/(self.obj_catalogue.total_rate)
                        objective_value1 = 0
                    
                    objective_value = float(cost)/jump_interval

                    objective.append(objective_value)

                    point_prop = self.cache.getAllPoints()

                    g_a = point_prop[1]
                    g_b = point_prop[2]

                    points_in_cache = point_prop[0]
                    number_points = len(points_in_cache)
                    
                    print("iter : ", i, "objective : ", objective_value, number_points, g_a, g_b)

                    self.write_stat(i, objective_value, f, number_points, objective_value1, g_a, g_b)                

                    self.write_stat_2(physical_hit, physical_miss, virtual_hit, virtual_miss, insert_new, real_update, f3)

                    if i < 100000 and i == 10 * jump_interval:
                        jump_interval *= 10
                    elif i == 100000 and i == 10 * jump_interval:
                        pass
                    elif i == 100 * jump_interval:
                        jump_interval *= 10

                    prev_i = i

                    if cache_initialized == True:
                        self.plot.plot_cache_pos_grid(points_in_cache, self.obj_catalogue.means, self.initial_points, count, [self.grid_d, self.grid_d], self.learning_rate, "virtual")
                        points_in_real_cache = [self.cache.obj_pos.virtual_to_real[v[0]][v[1]] for v in points_in_cache]
                        self.plot.plot_cache_pos_grid(points_in_real_cache, self.obj_catalogue.means, self.initial_points, count, [self.grid_d, self.grid_d], self.learning_rate, "real")
                    count += 1                

                    cost = 0


                if experiment_type != "gaussian_real":  
                    [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearest(pos)                        
                    new_object_loc = self.descent.descent(nearest_obj, obj)                                  
                    self.cache.updateCacheDict(nearest_obj, new_object_loc, mapped_x, mapped_y)                
                else :
                    if i < 4 * self.grid_d:                        
                        if len(self.cache.getAllPoints()[0]) < self.grid_d:
                            obj = self.obj_catalogue.getRequestGaussian2()
                            self.cache.insertInit(obj, i, policy)
                            continue
                        else:
                            cache_initialized = True

                    if experiment_type == "uniform" :
                        obj = self.obj_catalogue.getRequest()
                    elif experiment_type == "gaussian" :
                        obj = self.obj_catalogue.getRequestGaussian()
                    elif experiment_type == "gaussian2" :
                        obj = self.obj_catalogue.getRequestGaussian2()
                    elif experiment_type == "gaussian_real" :
                        obj = self.obj_catalogue.getRequestGaussian2()

                    pos = obj

                    #print("Requested object : ", obj)
                    [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearestVirtual(pos)                        

                    new_object_loc = self.descent.descent(nearest_obj, obj, "gaussian_real")                

                    ## Update the required tables after descending
                    self.cache.updateVirtualObjectAndFreq(nearest_obj, new_object_loc, mapped_x, mapped_y, i, policy)                
                    
                    #no_points = len(self.cache.getAllPoints())

                    if dst <= Threshold : ## Virtual hit

                        virtual_hit += 1

                        mapped_real_object = self.cache.getReal(new_object_loc)

                        ## Find the distance between the requested object and the object in cache which could serve the request
                        dist = l1_dist(pos, mapped_real_object)

                        ## Find the distance between the mapped real object to the virtual object
                        dst2 = l1_dist(np.array([mapped_real_object[0], mapped_real_object[1]]), np.array([nearest_obj[0], nearest_obj[1]]))

                        ## If the object in catalogue is the nearest real object to the requested object
                        if dist <= 1:                            
                            cost += dist
                        
                        ## If there is an object which is closer to the virtual object
                        elif dist <= Threshold : # physical hit
                            #print(" Physical hit ")

                            physical_hit += 1

                            ## With some probability insert the object never the less
                            check_if_in_real_cache = (obj[0] == mapped_real_object[0] and obj[1] == mapped_real_object[1])
                            if_better_object = dst < dst2

                            if random.random() < 0.1 and check_if_in_real_cache == False and if_better_object:                                
                                #print("Evicting and inserting a new object ", obj)
                                new_real_obj = obj
                                virtual_obj = new_object_loc#nearest_obj
                                orig_real_obj = mapped_real_object

                                real_update += 1

                                self.cache.updateRealObject(new_real_obj, i, policy, virtual_obj, orig_real_obj)
                                cost += Threshold 
                            else:
                                if random.random() < 0.01 :
                                    self.cache.insert(obj, i, policy)
                                    cost += Threshold
                                    insert_new += 1
                                else :
                                    cost += dist
                        else:
                            ## it is a virtual hit and a physical miss, so fetch the nearest object in the cache and insert
                            ## the object in the cache

                            nearest_catalogue = self.obj_catalogue.nearestObject(nearest_obj)
                            nearest_catalogue = np.array([nearest_catalogue[0], nearest_catalogue[1]])
                            
                            approx_cost = l1_dist(nearest_catalogue, obj)                            

                            if approx_cost <= Threshold:                                             
                                ## We should almost always end up here            
                                #print("Evicting and inserting a new object ", obj)                    
                                cost += Threshold
                                physical_hit += 1
                                #cost += approx_cost
                                #self.cache.insert(nearest_catalogue, i, policy)
                            else :
                                cost += Threshold
                                physical_miss += 1

                    else:
                        [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearestReal(pos)                        

                        if dst <= Threshold: ## physical hit and virtual miss:
                            cost += dst
                            physical_hit += 1
                        else :
                            cost += Threshold
                            if random.random() < 0.01 :
                                self.cache.insert(obj, i, policy)
                                insert_new += 1



                                                                                                            
#            if len(self.cache.getAllPoints()) > 313:
#                print("iter : ", i, "Request : ", obj, " Nearest : ", nearest_obj, " Mapped : ", mapped_x, mapped_y)
#                self.write_stat_debug(f2, obj, nearest_obj, mapped_x, mapped_y)
#                break
                
        f2 = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type + '_' + file_name_extension + '/distances.txt', 'w')
        self.write_distance_count(self.obj_catalogue.obj_count_distance, f2)
        f2.write(str(len(self.cache.getAllPoints()[0])))
        f2.close()

s = Simulator(2, 313, 100, 0.4, 100000000, 1, 0.01)
s.simulate()                



                



        
