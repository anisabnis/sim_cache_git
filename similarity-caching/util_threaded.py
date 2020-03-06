""" This file contains the various utilities which can be used to simulate a similarity cache """

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import operator
from decimal import *
from collections import defaultdict
import multiprocessing
import random
getcontext().prec = 2

""" All attributes of a cache object """
class CacheObject:
    counter = 0

    def __init__(self, id, vec, center=[0,0], rho=0, size=1):
        CacheObject.counter += 1
        self.id = CacheObject.counter
        self.pos = vec
        dist = np.linalg.norm(vec - center, ord=1)
        self.rate = np.exp(-(dist ** 2)/rho)
        self.size = size


class objPos:
    def __init__(self, grid_s):
        self.cache = defaultdict(lambda : defaultdict(list))
        self.grid = grid_s

    def checkIfInGrid(self, v):
        if v[0] >= 0 and v[0] <= self.grid[0] and v[1] >= 0 and v[1] <= self.grid[1]:
            return True
        else:
            return False


    def findOriginalPoint(self, v):
        x = 0
        y = 0

        if v[0] >= self.grid[0]:
            x = round(v[0] - self.grid[0],3)
        elif v[0] <= 0:
            x = round(v[0] + self.grid[0],3)
        else :
            x = round(v[0],3)

        if v[1] >= self.grid[1]:
            y = round(v[1] - self.grid[1],3)
        elif v[1] <= 0:
            y = round(v[1] + self.grid[1],3)
        else :
            y = round(v[1],3)

        return np.array([x,y])
        
    def insert(self, point):
        def check(test,array):
            return any(np.array_equal(x, test) for x in array)

        if self.checkIfInGrid(point) == True:
            if check(point, self.cache[int(point[0])][int(point[1])]) == True:
                pass
            else:
                self.cache[int(point[0])][int(point[1])].append(point) 
        else :
            new_point = self.findOriginalPoint(point)
            if check(new_point, self.cache[int(new_point[0])][int(new_point[1])]) == True:
                pass
            else :
                self.cache[int(new_point[0])][int(new_point[1])].append(new_point) 

    def delete(self, point, mapped):
        if mapped == False:
            self.cache[int(point[0])][int(point[1])] = [x for x in list(self.cache[int(point[0])][int(point[1])]) if x[0] != point[0] and x[1] != point[1]]
        else:
            new_point = self.findOriginalPoint(point)
            self.cache[int(new_point[0])][int(new_point[1])] = [x for x in list(self.cache[int(new_point[0])][int(new_point[1])]) if x[0] != new_point[0] and x[1] != new_point[1]]

    def get_all_points(self):
        points = []
        for a in self.cache:
            for b in self.cache[a]:
                points.extend(self.cache[a][b])
        return points
                                                    
        
""" All attributes and functions of a cache """
class CacheGrid:
    def __init__(self, capacity, dim, learning_rate, integral=False, grid_s=[313,313]):

        self.cache = {}
        self.grid = grid_s
        self.integral = integral

        f = open("cache_pos.txt", "r")

        for index in range(capacity):
            l = f.readline()
            l = l.strip().split(" ")
            if integral == False:
                v = np.random.rand(dim)
                self.cache[index] = v
            else:
                ii = int(l[0])
                jj = int(l[1])
                self.cache[index] = np.array([ii,jj])

        self.alpha = learning_rate
        self.capacity = capacity
        self.initializeIterativeSearch(self.grid)

    def getAllPoints(self):
        return self.obj_pos.get_all_points()
    
    def initializeIterativeSearch(self, dim):
        self.obj_pos = objPos(self.grid)
        for index in range(self.capacity):
            self.obj_pos.insert(self.cache[index])
                                        
    def updateCache(self, src_obj_pos, dst_obj, mapped):
        self.obj_pos.delete(src_obj_pos, mapped)
        self.obj_pos.insert(dst_obj)
        
    def updateCacheDict(self, src_obj_pos, dst_obj, mapped):
        self.updateCache(src_obj_pos, dst_obj, mapped)

    def get_mapped_points(self, v):
        if v[0] <= float(self.grid[0]/2) and v[0] >= 0:
            if v[1] <= float(self.grid[1]/2) and v[1] >= 0:
                first_point = np.array([round(v[0] + self.grid[0], 3), round(v[1],3)])
                second_point = np.array([round(v[0], 3), round(v[1] + self.grid[1],3)])
                third_point = np.array([round(v[0] + self.grid[0],3), round(v[1] + self.grid[1],3)])
            else:
                if v[1] > float(self.grid[1]/2)  and v[1] <= self.grid[1]:
                    first_point = np.array([round(v[0] + self.grid[0],3), round(v[1],3)])
                    second_point = np.array([round(v[0],3), round(v[1] - self.grid[1],3)])
                    third_point = np.array([round(v[0] + self.grid[0],3), round(v[1] - self.grid[1],3)])
        else :
            if v[0] > float(self.grid[0]/2) and v[0] <= self.grid[0]:
                if v[1] <= float(self.grid[1]/2) and v[1] >= 0:
                    first_point = np.array([round(v[0] - self.grid[0],3), round(v[1],3)])
                    second_point = np.array([round(v[0],3) , round(v[1] + self.grid[1],3)])
                    third_point = np.array([round(v[0] - self.grid[0],3), round(v[1] + self.grid[1],3)])
                else:
                    if v[1] > float(self.grid [1]/2) and v[1]  <= self.grid[1]:
                        first_point = np.array([round(v[0] - self.grid[0],3), round(v[1],3)])
                        second_point = np.array([round(v[0],3), round(v[1] - self.grid[1],3)])
                        third_point = np.array([round(v[0] - self.grid[0],3), round(v[1] - self.grid[1],3)])

        return [first_point, second_point, third_point]

    def findNearest(self, vec):
        i = 0
        min_dist = 10000
        min_point = [0,0]
        found = False
        candidates = []        
        break_i = self.grid[0]
        first = True

        while i <= break_i:

            x1 = (int(vec[0])-i)%self.grid[0]
            x2 = (int(vec[0])+i)%self.grid[0]

            y1 = (int(vec[1])-i)%self.grid[1]
            y2 = (int(vec[1])+i)%self.grid[1]

            a = i
            for x in range(int(vec[0])-i, int(vec[0]) + i + 1):
                x = x%self.grid[0]
                
                if first == True or (first == False and abs(a) + i <= break_i):
               
                    candidates.extend(self.obj_pos.cache[x][y1])
                    candidates.extend(self.obj_pos.cache[x][y2])
                   
                    if first == True:
                        if len(self.obj_pos.cache[x][y1]) > 0:
                            if found == False:
                                found = True
                       
                        if len(self.obj_pos.cache[x][y2]) > 0:
                            if found == False:
                                found = True
                a=a-1
                        
            a = i
            for y in range(int(vec[1])-i, int(vec[1]) + i + 1):
                y = y%self.grid[1]
                
                if first == True or (first == False and abs(a) + i <= break_i):
                    candidates.extend(self.obj_pos.cache[x1][y])
                    candidates.extend(self.obj_pos.cache[x2][y])            

                    if first == True:
                        if len(self.obj_pos.cache[x1][y]) > 0:
                            if found == False:
                                found = True
                            
                        if len(self.obj_pos.cache[x2][y]) > 0:
                            if found == False:
                                found = True                    

                a=a-1
                
            if found == True and first == True:
                break_i = math.ceil(i * 2) + 1
                first = False

            i += 1


        def dist(c, v, break_i):
            first = np.linalg.norm((c-v), ord=1)
            if first > 4 * break_i:
                mapped_points = self.get_mapped_points(c)
                mapped = [(c, np.linalg.norm((c-v), ord=1)) for c in mapped_points]
                best = min(mapped, key=operator.itemgetter(1))                
                return [best[0], best[1], True]
            else:
                return [c , first, False]

        candidates = [dist(c, vec, break_i) for c in candidates]
        best_candidate = min(candidates, key=operator.itemgetter(1))
        return [best_candidate[0], best_candidate[1], best_candidate[2]]


""" Object Catalogue """
class ObjectCatalogueUniform:
    def __init__(self, no_objects, alpha, dim):
        
        self.catalogue = []

        ## generate objects at random locations        
        for i in range(no_objects):
            loc = np.random.rand(dim)
            c_obj = CacheObject(i, loc)
            self.catalogue.append(c_obj)

        N = no_objects
        x = np.arange(1, N)
        a = alpha
        weights = x ** (-a)
        weights /= weights.sum()
        self.bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))

    def objective(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            min_dst = 10
            for c in cache:
                if np.linalg.norm(c_obj.pos - cache[c]) < min_dst:
                    min_dst =  np.linalg.norm(c_obj.pos - cache[c])
            obj += min_dst
        return obj
    
    def objective_l1(self, cache):
        return 0

    def getRequest(self):
        obj = self.catalogue[self.bounded_zipf.rvs(size=1)[0]]
        return obj
      

class ObjectCatalogueGrid:
    def __init__(self, dim_x, dim_y):
        self.catalogue = []
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.means = []
        self.obj_id = 0
        self.center = np.array([float(dim_x)/2, float(dim_y)/2])
        self.rho = (self.dim_x ** 2)/64 #39.125 *  2 * 39.125
        self.rho *= 2
        self.obj_count_distance = defaultdict(int)

        for i in range(dim_x):
            for j in range(dim_y):

                pos = np.array([i,j])
                c_obj = CacheObject(self.obj_id, pos, self.center, self.rho)                
                self.catalogue.append(c_obj)
                self.means.append(pos)
                self.obj_id += 1
                
    def getRequest(self):
        index = np.random.randint(0, self.obj_id)
        obj = self.catalogue[index]
        return obj
        
    def getRequestGaussian(self):
        iter = 0
        while 1:
            i = np.random.randint(self.dim_x + 1)
            j = np.random.randint(self.dim_y + 1)
            point = np.array([i,j])
            distance = np.linalg.norm(point - self.center, ord=1)
            acc_distance = int(distance)
            distance = float(distance ** 2)/self.rho

            r_no = random.random()
            if r_no < np.exp(-distance):
                self.obj_count_distance[int(acc_distance)] += 1            
                return point
            iter += 1
                
    def objective(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            min_dst = 100000
            for c in cache:
                if np.linalg.norm(c_obj.pos - c) < min_dst:
                    min_dst =  np.linalg.norm(c_obj.pos - c)
            obj += min_dst
        return obj


    def objective_l1_lsh(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            K = cache.lsh.query(c_obj.pos)
            obj += K[1]            
        return obj

    def objective_l1_iterative(self, sub_catalogue, objective_dict, i, cache, t):
        obj = 0
        if t == "uniform":
            for c_obj in sub_catalogue:
                K = cache.findNearest(c_obj.pos)
                obj += K[1]
            objective_dict[i] = obj
            return
        else :
            for c_obj in sub_catalogue:
                K = cache.findNearest(c_obj.pos)
                obj += (K[1] * c_obj.rate)
            objective_dict[i] = obj
            return    


    def objective_l1_iterative_threaded(self, cache, t="uniform"):
        manager = multiprocessing.Manager()
        objective_val = manager.dict()
        obj = 0

        sequence = list(range(8))
        sequence.reverse()
        chunk_size = int(len(self.catalogue)/8)

        jobs = []
        p = multiprocessing.Process(target=self.objective_l1_iterative, args=(self.catalogue[sequence[0]*chunk_size:], objective_val, 8, cache, t,))
        p.start()
        jobs.append(p)        

        for i in range(1, len(sequence)):
            p = multiprocessing.Process(target=self.objective_l1_iterative, args=(self.catalogue[sequence[i]*chunk_size:sequence[i-1]*chunk_size], objective_val, sequence[i], cache, t, ))

            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        obj = sum(objective_val.values())
        return obj

    
""" Object Catalogue """
class ObjectCatalogueGaussian:
    def __init__(self, no_objects, centers, dim):
        
        self.catalogue = []
        self.means = []
        self.covs = []
        self.no_objects = no_objects

        for i in range(centers):
            v = np.random.uniform(low=0.1, high=0.9, size=(dim,))
            c = new_random_cov(dim)
            self.means.append(v)
            self.covs.append(c)

        for i in range(no_objects):
            index = np.random.randint(0, centers)
            loc = np.random.multivariate_normal(self.means[index], self.covs[index])
            c_obj = CacheObject(i, loc)
            self.catalogue.append(c_obj)

    def getRequest(self):
        index = np.random.randint(0, self.no_objects)
        obj = self.catalogue[index]
        return obj

    def objective(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            min_dst = 10
            for c in cache:
                if np.linalg.norm(c_obj.pos - cache[c]) < min_dst:
                    min_dst =  np.linalg.norm(c_obj.pos - cache[c])
            obj += min_dst
        return obj

""" DescentMethods """
class StochasticGradientDescent:
    def __init__(self, learning_rate, grid):
        self.alpha = learning_rate
        self.grid = grid

    def descent(self, nearest_object, current_object):

        def derivative_l2(nearest_object, current_object):
            return 2 * (nearest_object - current_object)

        def derivative_l1(nearest_object, current_object):            
            return np.array([-1 if nearest_object[i] - current_object[i] < 0 else 1 for i in range(len(nearest_object))])

        d = derivative_l1(nearest_object, current_object)
        n = nearest_object - self.alpha * d
        n = [x%self.grid for x in n]
        n = [round(x,3) for x in n]
        return n


""" Plot generation code """
class Plots:
    def __init__(self):
        pass

    def plot(self, time_series, grid_d, learning_rate):
        plt.plot(time_series)
        plt.ylabel("Objective")
        plt.xlabel("iterations")
        plt.grid()
        plt.savefig(str(grid_d) + "_" + str(learning_rate) + "/objective.png")
        plt.clf()

    def plot_cache_pos(self, cache, obj_means, cache_init):
        cache_objs = list(cache.values())
        xs = [l[0] for l in cache_objs]
        ys = [l[1] for l in cache_objs]
        plt.scatter(xs, ys, marker='+', label="cache")
        
        obj_cata = obj_means
        xs = [l[0] for l in obj_cata]
        ys = [l[1] for l in obj_cata]
        plt.scatter(xs, ys, marker='*', label="obj_mean")

        cache_objs = list(cache_init.values())
        xs = [l[0] for l in cache_objs]
        ys = [l[1] for l in cache_objs]
        plt.scatter(xs, ys, marker='o', label="initial")
        plt.legend()
        plt.show()
        plt.clf()
        


    def checkIfInGrid(self, v, grid):
        if v[0] >= 0 and v[0] <= grid[0] and v[1] >= 0 and v[1] <= grid[1]:
            return True
        else:
            return False

    def plot_cache_pos_grid(self, cache, obj_means, cache_init, count, grid, learning_rate):
        cache_objs = cache
        cache_objs = [v for v in cache_objs if self.checkIfInGrid(v, grid) == True]
        xs = [l[0] for l in cache_objs]
#        xs = [x for x in xs if x >=0 and x <= grid[0]]
        ys = [l[1] for l in cache_objs]
#        ys = [y for y in ys if y >= 0 and y <= grid[1]]
        plt.scatter(xs, ys, marker='+', label="cache")
        
        obj_cata = obj_means
        xs = [l[0] for l in obj_cata]
        ys = [l[1] for l in obj_cata]
        #plt.scatter(xs, ys, marker='*', label="obj_mean")

        cache_objs = cache_init #list(cache_init.values())
        xs = [l[0] for l in cache_objs]
        ys = [l[1] for l in cache_objs]
        #plt.scatter(xs, ys, marker='o', label="initial")

        plt.legend()
        plt.savefig(str(grid[0]) + "_" + str(learning_rate) + "_gaussian/cache_pos" + str(count) + ".png")
        plt.clf()
        
        


            


