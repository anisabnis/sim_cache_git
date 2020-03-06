from util import * 
import argparse
import copy
import os
import time

class Simulator:
    def __init__(self, dim, capacity, no_objects, alpha, iter, update_interval, learning_rate):
        #self.obj_catalogue = ObjectCatalogueGaussian(no_objects, 2, dim)
        #self.obj_catalogue = ObjectCatalogueUniform(no_objects, 0.8, dim)
        self.grid_d = 313

        self.obj_catalogue = ObjectCatalogueGrid(self.grid_d, self.grid_d)        

        self.cache = CacheGrid(capacity, dim, learning_rate, True, [self.grid_d, self.grid_d])
 
        self.iter = iter

        self.u_interval = update_interval

        self.descent = StochasticGradientDescent(learning_rate, self.grid_d)

        self.plot =  Plots()

        self.initial_points = self.cache.getAllPoints()    

        self.learning_rate = learning_rate

        os.system("mkdir " + str(self.grid_d) + "_" + str(learning_rate) + "_fixcache_2")        
        

    def write_stat(self, i, obj, f):
        f.write(str(i) + "\t" + str(obj))
        f.write("\n")
        f.flush()
                

    def simulate(self):
        objective = [] 
        objective_value = 0
        
        count = 0
        prev_i = 0
        jump_interval = 1

        number_obj = len(self.cache.getAllPoints())

        f = open(str(self.grid_d) + '_' + str(self.learning_rate) + '_fixcache_2' +  '/' + str("objective") + '.txt', 'w')                
               
        for i in range(self.iter):

            obj = self.obj_catalogue.getRequest()
            pos = obj.pos

            [nearest_obj, dst, mapped] = self.cache.findNearest(pos)                        

            if i % self.u_interval == 0:
                new_object_loc = self.descent.descent(nearest_obj, obj.pos)
                self.cache.updateCacheDict(nearest_obj, new_object_loc, mapped)                

            if i - prev_i >= jump_interval:

                objective_value = self.obj_catalogue.objective_l1_iterative_threaded(self.cache)                

                objective.append(objective_value)

                self.write_stat(i, objective_value, f)

                if i < 100000 and i == 10 * jump_interval:
                    jump_interval *= 10
                elif i == 100000 and i == 10 * jump_interval:
                    pass
                elif i == 100 * jump_interval:
                    jump_interval *= 10

                prev_i = i

                self.plot.plot_cache_pos_grid(self.cache.getAllPoints(), self.obj_catalogue.means, self.initial_points, count, [self.grid_d, self.grid_d], self.learning_rate)
                count += 1                


s = Simulator(2, 313, 100, 0.4, 150000000, 1, 0.001)
s.simulate()                



                



        
