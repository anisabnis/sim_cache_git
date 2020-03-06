from util import *
import argparse
import copy
import os
import time
import sys

experiment_type = sys.argv[1]
file_name_extension = sys.argv[2]

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
        self.grid_d = 313

        self.obj_catalogue = ObjectCatalogueGrid(self.grid_d, self.grid_d, experiment_type)        

        self.cache = CacheGrid(capacity, dim, learning_rate, True, [self.grid_d, self.grid_d])

        self.cache.initializeIterativeSearch([self.grid_d, self.grid_d])
 
        self.iter = iter

        self.u_interval = update_interval

        self.descent = StochasticGradientDescent(learning_rate, self.grid_d)

        self.plot =  Plots(experiment_type, file_name_extension)

        self.initial_points = self.cache.getAllPoints()    

        self.learning_rate = learning_rate

        os.system("mkdir " + str(self.grid_d) + "_" + str(learning_rate) + "_" + experiment_type + "_" + file_name_extension)        
        
        self.cache_capacity = capacity

    def write_stat(self, i, obj, f, cache_size):
        f.write(str(i) + "\t" + str(obj) + "\t" + str(cache_size))
        f.write("\n")
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
        jump_interval = 1

        number_obj = len(self.cache.getAllPoints())

        f = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '/' + str("objective") + '.txt', 'w')                
        f2 = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type +  '_' + file_name_extension + '/' + str("debug") + '.txt', 'w')                               
        cost = 0

        Threshold = 10



        for i in range(1,self.iter):

            if experiment_type == "uniform" :
                obj = self.obj_catalogue.getRequest()
            elif experiment_type == "gaussian" :
                obj = self.obj_catalogue.getRequestGaussian()
            elif experiment_type == "gaussian2" :
                obj = self.obj_catalogue.getRequestGaussian2()
            elif experiment_type == "gaussian_real" :
                obj = self.obj_catalogue.getRequestGaussian2()

            pos = obj

            if i % self.u_interval == 0:                

                if experiment_type != "gaussian_real":  
                    [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearest(pos)                        
                    new_object_loc = self.descent.descent(nearest_obj, obj)                               
                    self.cache.updateCacheDict(nearest_obj, new_object_loc, mapped_x, mapped_y)                

                else :
                    if i < 4 * self.grid_d:                        
                        if len(self.cache.getAllPoints()) < self.grid_d:
                            self.cache.insertInit(obj)
                            continue

                    #print("Initial objects in cache : ", self.cache.freq_map[0])

                    #print("Requested object : ", obj)

                    [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearestVirtual(pos)                        

                    #print("Nearest Virtual Object : ", nearest_obj)

                    new_object_loc = self.descent.descent(nearest_obj, obj, "gaussian_real")                

                    #print("New Object Location : ", new_object_loc)
                    
                    ## Update the required tables after descending
                    self.cache.updateVirtualObjectAndFreq(nearest_obj, new_object_loc, mapped_x, mapped_y)                
                    
                    if dst <= Threshold : ## Virtual hit
                        #print(" It is a virtual hit ")
                        mapped_real_object = self.cache.getReal(new_object_loc)

                        #print(" Mapped real object : ", mapped_real_object)
                        dist = l1_dist(pos, mapped_real_object)

                        ## If the object in catalogue is the nearest real object to the virtual object
                        if dist <= 1:                            
                            cost += dist
                        
                        ## If there is an object which is closer to the virtual object
                        elif dist <= Threshold : # physical hit
                            #print(" Physical hit ")

                            ## With some probability insert the object never the less
                            check_if_in_real_cache = (obj[0] == mapped_real_object[0] and obj[1] == mapped_real_object[1])
                            dst2 = l1_dist(np.array([mapped_real_object[0], mapped_real_object[1]]), np.array([nearest_obj[0], nearest_obj[1]]))
                            if_better_object = dst < dst2

                            if random.random() < 0.02 and check_if_in_real_cache == False and if_better_object:                                
                                self.cache.insert(obj)
                                cost += Threshold 
                            else:
                                cost += l1_dist(mapped_real_object, obj)
                        else:
                            ## it is a virtual hit and a physical miss, so fetch the nearest object in the cache and insert
                            ## the object in the cache

                            nearest_catalogue = self.obj_catalogue.nearestObject(nearest_obj)
                            nearest_catalogue = np.array([nearest_catalogue[0], nearest_catalogue[1]])
                            
                            approx_cost = l1_dist(nearest_catalogue, obj)                            

                            if approx_cost <= Threshold:                                             
                                ## We should almost always end up here                                
                                cost += approx_cost
                                self.cache.insert(nearest_catalogue)
                            else :
                                cost += Threshold

                    else:
                        [nearest_obj, dst, mapped_x, mapped_y] = self.cache.findNearestReal(pos)                        

                        if dst <= Threshold: ## physical hit and virtual miss:
                            cost += dst
                        else :
                            cost += Threshold

                
                if i - prev_i >= jump_interval:
                    
                    objective_value = self.obj_catalogue.objective_l1_iterative_threaded(self.cache, experiment_type)                
                    objective_value = objective_value/(self.obj_catalogue.total_rate)
                    
                    #objective_value = float(cost)/i

                    objective.append(objective_value)

                    print("iter : ", i, "objective : ", objective_value)

                    self.write_stat(i, objective_value, f, len(self.cache.getAllPoints()))                

                    if i < 100000 and i == 10 * jump_interval:
                        jump_interval *= 10
                    elif i == 100000 and i == 10 * jump_interval:
                        pass
                    elif i == 100 * jump_interval:
                        jump_interval *= 10

                    prev_i = i

                     #self.plot.plot_cache_pos_grid(self.cache.getAllPoints(), self.obj_catalogue.means, self.initial_points, count, [self.grid_d, self.grid_d], self.learning_rate)
                    count += 1                

#                    cost = 0

                                                                                                            
#            if len(self.cache.getAllPoints()) > 313:
#                print("iter : ", i, "Request : ", obj, " Nearest : ", nearest_obj, " Mapped : ", mapped_x, mapped_y)
#                self.write_stat_debug(f2, obj, nearest_obj, mapped_x, mapped_y)
#                break
                
        f2 = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type + '_' + file_name_extension + '/distances.txt', 'w')
        self.write_distance_count(self.obj_catalogue.obj_count_distance, f2)
        f2.write(str(len(self.cache.getAllPoints())))
        f2.close()

s = Simulator(2, 313, 100, 0.4, 100000000, 1, 0.01)
s.simulate()                



                



        
