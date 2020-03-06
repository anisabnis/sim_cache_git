from util import * 
import argparse
import copy
import os
import time
import sys

experiment_type = sys.argv[1]

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
        self.grid_d = 313

        self.obj_catalogue = ObjectCatalogueGrid(self.grid_d, self.grid_d)        

        self.cache = CacheGrid(capacity, dim, learning_rate, True, [self.grid_d, self.grid_d])
 
        self.iter = iter

        self.u_interval = update_interval

        self.descent = StochasticGradientDescent(learning_rate, self.grid_d)

        self.plot =  Plots(experiment_type)

        self.initial_points = self.cache.getAllPoints()    

        self.learning_rate = learning_rate

        os.system("mkdir " + str(self.grid_d) + "_" + str(learning_rate) + "_" + experiment_type)        
        

    def write_stat(self, i, obj, f):
        f.write(str(i) + "\t" + str(obj))
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

    def simulate(self):
        objective = [] 
        objective_value = 0
        
        count = 0
        prev_i = 0
        jump_interval = 1

        number_obj = len(self.cache.getAllPoints())

        f = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type +  '/' + str("objective") + '.txt', 'w')                
               
        for i in range(self.iter):

            obj = self.obj_catalogue.getRequestGaussian()
            pos = obj

            [nearest_obj, dst, mapped] = self.cache.findNearest(pos)                        

            if i % self.u_interval == 0:
                new_object_loc = self.descent.descent(nearest_obj, obj)
                self.cache.updateCacheDict(nearest_obj, new_object_loc, mapped)                

            if i - prev_i >= jump_interval:

                objective_value = self.obj_catalogue.objective_l1_iterative_threaded(self.cache, experiment_type)                

                objective.append(objective_value)

                self.write_stat(i, objective_value, f)
                
                print("iter : ", i, "objective : ", objective_value)

                if i < 100000 and i == 10 * jump_interval:
                    jump_interval *= 10
                elif i == 100000 and i == 10 * jump_interval:
                    pass
                elif i == 100 * jump_interval:
                    jump_interval *= 10

                prev_i = i

                self.plot.plot_cache_pos_grid(self.cache.getAllPoints(), self.obj_catalogue.means, self.initial_points, count, [self.grid_d, self.grid_d], self.learning_rate)
                count += 1                

                
        f2 = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_' + experiment_type + '/distances.txt', 'w')
        self.write_distance_count(self.obj_catalogue.obj_count_distance, f2)


s = Simulator(2, 313, 100, 0.4, 150000000, 1, 0.01)
s.simulate()                



                



        
