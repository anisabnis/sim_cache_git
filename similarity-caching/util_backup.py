""" This file contains the various utilities which can be used to simulate a similarity cache """

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
from lshash import *
from decimal import *
from collections import defaultdict
import multiprocessing
getcontext().prec = 2

# new random covar
def new_random_cov(n):
    if n == 1:
        return random.uniform(0.05, 0.1)

    # random matrix generation
    A = []
    for i in range(0, n):
        A.append([])
        for j in range(0, n):
            A[i].append([])

    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = np.random.triangular(-0.1, 0, 0.1)

    A = np.array(A)
    B = np.matmul(A, A.transpose())
    C = np.zeros((n, n))

    for i in range(0, n):
        diago = 1 / math.sqrt(B[i][i])
        C[i][i] = diago
    first_mol = np.matmul(C, B)
    C = np.matmul(first_mol, C)
    D = np.zeros((n, n))

    for i in range(0, n):
        diago = np.random.uniform(0.05, 0.1)
        D[i][i] = diago
    first_mol = np.matmul(D, C)
    res = np.matmul(first_mol, D)
    return res


""" All attributes of a cache object """
class CacheObject:
    counter = 0

    def __init__(self, id, vec, size=1):
        CacheObject.counter += 1
        self.id = CacheObject.counter
        self.pos = vec
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

        if v[0] > self.grid[0]:
            x = round(v[0] - self.grid[0],3)
        elif v[0] < 0:
            x = round(v[0] + self.grid[0],3)
        else :
            x = round(v[0],3)

        if v[1] > self.grid[1]:
            y = round(v[1] - self.grid[1],3)
        elif v[1] < 0:
            y = round(v[1] + self.grid[1],3)
        else :
            y = round(v[1],3)

        return np.array([x,y])
        
    def insert(self, point):
        if self.checkIfInGrid(point) == True:
            if point in self.cache[int(point[0])][int(point[1])]:
                pass
            else:
                self.cache[int(point[0])][int(point[1])].append(point) 
        else :
            new_point = self.findOriginalPoint(point)
            if new_point in self.findOriginalPoint(point):
                pass
            else :
                self.cache[int(new_point[0])][int(new_point[1])].append(new_point) 

    def delete(self, point, mapped):
        #if self.checkIfInGrid(point) == True:
        if mapped == False:
#            print("Deleting point : ", point, self.cache[int(point[0])][int(point[1])])
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

        for index in range(capacity):
            if integral == False:
                v = np.random.rand(dim)
                self.cache[index] = v
            else:
                ii = np.random.randint(0, grid_s[0])
                jj = np.random.randint(0, grid_s[1])
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

        return [v, first_point, second_point, third_point]

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
                break_i = math.ceil(i * 2)
                first = False

            i += 1

        ## Returns [a, c]
        ## a = mapped point
        ## c = distance
        def dist(c, v, break_i):
            min_dst = 10000
            first = np.linalg.norm((c-v), ord=1)
            if first > 4 * break_i:
                mapped_points = self.get_mapped_points(c)
                mapped = [(c, np.linalg.norm((c-v), ord=1)) for c in mapped_points]
                best = min(mapped, key=operator.itemgetter(1))
                if best != c:
                    return [best[0], best[1], True]
                else :
                    return [best[0], best[1], False]
            else:
                return [c , first, False]

        candidates = [dist(c, vec, break_i) for c in candidates]
        best_candidate = min(candidates, key=operator.itemgetter(1))
        return [best_candidate[0], best_candidate[1], best_candidate[2]]

            
""" All attributes and functions of a cache """
class Cache:
    def __init__(self, capacity, dim, learning_rate, integral=False, grid_s=[313,313]):
        #self.cache = np.random.rand(capacity, dim)

        self.cache = {}
        self.grid = grid_s
        self.integral = integral

        for index in range(capacity):
            if integral == False:
                v = np.random.rand(dim)
                self.cache[index] = v
            else:
                ii = np.random.randint(0, grid_s[0])
                jj = np.random.randint(0, grid_s[1])
                self.cache[index] = np.array([ii,jj])

        self.alpha = learning_rate
        self.capacity = capacity
        self.initializeLSH(dim)        

    def getAllPoints(self):
        return self.lsh.get_all_points()


    def extraPoints(self, v, delete=False):
        update = False
        if v[0] <= float(self.grid[0]/2) and v[0] >= 0:
            if v[1] <= float(self.grid[1]/2) and v[1] >= 0:
                first_point = np.array([round(v[0] + self.grid[0], 3), round(v[1],3)])
                second_point = np.array([round(v[0], 3), round(v[1] + self.grid[1],3)])
                third_point = np.array([round(v[0] + self.grid[0],3), round(v[1] + self.grid[1],3)])
                update = True
            else:
                if v[1] > float(self.grid[1]/2)  and v[1] <= self.grid[1]:
                    first_point = np.array([round(v[0] + self.grid[0],3), round(v[1],3)])
                    second_point = np.array([round(v[0],3), round(v[1] - self.grid[1],3)])
                    third_point = np.array([round(v[0] + self.grid[0],3), round(v[1] - self.grid[1],3)])
                    update = True
        else :
            if v[0] > float(self.grid[0]/2) and v[0] <= self.grid[0]:
                if v[1] <= float(self.grid[1]/2) and v[1] >= 0:
                    first_point = np.array([round(v[0] - self.grid[0],3), round(v[1],3)])
                    second_point = np.array([round(v[0],3) , round(v[1] + self.grid[1],3)])
                    third_point = np.array([round(v[0] - self.grid[0],3), round(v[1] + self.grid[1],3)])
                    update = True
                else:
                    if v[1] > float(self.grid [1]/2) and v[1]  <= self.grid[1]:
                        first_point = np.array([round(v[0] - self.grid[0],3), round(v[1],3)])
                        second_point = np.array([round(v[0],3), round(v[1] - self.grid[1],3)])
                        third_point = np.array([round(v[0] - self.grid[0],3), round(v[1] - self.grid[1],3)])
                        update = True

        
        if update == False:
            return 
        elif delete == False:
            self.lsh.index(first_point)
            self.lsh.index(second_point)
            self.lsh.index(third_point)
        else:
            self.lsh.delete_vector(first_point)
            self.lsh.delete_vector(second_point)
            self.lsh.delete_vector(third_point)                        

            
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

    
    def initializeLSH(self, dim):        
        self.lsh = LSHash(4, dim, 6)
        for index in range(self.capacity):
            v = self.cache[index]
            if self.integral == False:
                self.lsh.index(v)
            else:
                self.lsh.index(v)
                self.extraPoints(v)
                
    def updateCache(self, src_obj_pos, dst_obj):
        ## Delete the point
        if self.checkIfInGrid(src_obj_pos) == True:
            self.lsh.delete_vector(src_obj_pos)
            self.extraPoints(src_obj_pos, True)
        else :
            v = self.findOriginalPoint(src_obj_pos)
            self.lsh.delete_vector(v)
            self.extraPoints(v, True)

        ## Add the new point            
        if self.checkIfInGrid(dst_obj.pos) == True:
            self.lsh.index(dst_obj.pos)
            self.extraPoints(dst_obj.pos)
        else:
            v = self.findOriginalPoint(dst_obj.pos)
            self.lsh.index(v)
            self.extraPoints(v)

    
    def updateCacheDict(self, src_obj_pos, dst_obj):
        self.updateCache(src_obj_pos, dst_obj)

    def findNearest(self, vec):
        nearest_point = []
        min_dst = 10000
        min_id = 0

        for index in self.cache:
            if np.linalg.norm(self.cache[index] - vec) < min_dst:
                nearest_point = self.cache[index]
                min_dst = np.linalg.norm(self.cache[index] - vec)
                min_id = index

        return [nearest_point, min_id]            
            

    def findNearestANN(self, vec):
        K = self.lsh.query(vec)
        nearest_point = K[0]
        min_dst = K[1]
        return [nearest_point, min_dst]


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

        for i in range(dim_x):
            for j in range(dim_y):

                pos = np.array([i,j])
                c_obj = CacheObject(self.obj_id, pos)                
                self.catalogue.append(c_obj)
                self.means.append(pos)
                self.obj_id += 1
                
    def getRequest(self):
        index = np.random.randint(0, self.obj_id)
        obj = self.catalogue[index]
        return obj
        

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

    def objective_l1_iterative(self, sub_catalogue, objective_dict, i, cache):
        obj = 0
        for c_obj in sub_catalogue:
            K = cache.findNearest(c_obj.pos)
            obj += K[1]
        objective_dict[i] = obj
        return 

    def objective_l1_iterative_threaded(self, cache):
        manager = multiprocessing.Manager()
        objective_val = manager.dict()
        obj = 0

        sequence = list(range(8))
        sequence.reverse()
        chunk_size = int(len(self.catalogue)/8)

        jobs = []
        p = multiprocessing.Process(target=self.objective_l1_iterative, args=(self.catalogue[sequence[0]*chunk_size:], objective_val, 8, cache, ))
        p.start()
        jobs.append(p)
        
        for i in range(1, len(sequence)):
            p = multiprocessing.Process(target=self.objective_l1_iterative, args=(self.catalogue[sequence[i]*chunk_size:sequence[i-1]*chunk_size], objective_val, sequence[i], cache, ))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()
            
        obj = sum(objective_val.values())
        return obj

    

class ObjectCatalogueGrid2:
    def __init__(self, dim_x, dim_y):
        self.catalogue = []
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.means = []
        self.obj_id = 0

        for i in range(dim_x):
            for j in range(dim_y):

                pos = np.array([i,j])
                c_obj = CacheObject(self.obj_id, pos)                
                self.catalogue.append(c_obj)
                self.means.append(pos)
                self.obj_id += 1
                
    def getRequest(self):
        index = np.random.randint(0, self.obj_id)
        obj = self.catalogue[index]
        return obj
        

    def objective(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            min_dst = 100000
            for c in cache:
                if np.linalg.norm(c_obj.pos - c) < min_dst:
                    min_dst =  np.linalg.norm(c_obj.pos - c)
            obj += min_dst
        return obj


    def objective_l1(self, cache):
        obj = 0
        for c_obj in self.catalogue:
            K = cache.lsh.query(c_obj.pos)
            obj += K[1]            
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
        plt.savefig(str(grid[0]) + "_" + str(learning_rate) + "_latest/cache_pos" + str(count) + ".png")
        plt.clf()
        
        


            


