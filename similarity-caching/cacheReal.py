from util import *
from LFU import *
import bisect
import heapq

import copy

class objPosVirtual:
    def __init__(self, grid_s):
        self.cache = defaultdict(lambda : defaultdict(list))
        self.virtual_to_real = defaultdict(lambda : defaultdict())
        self.real_to_virtual = defaultdict(lambda : defaultdict())        
        self.grid = grid_s

    def checkIfInGrid(self, v):

        if v[0] >= 0 and v[0] <= self.grid[0] and v[1] >= 0 and v[1] <= self.grid[1]:
            return True
        else:
            return False

    def findOriginalPoint(self, v, mapped_x, mapped_y):
        x = 0
        y = 0

        if v[0] >= self.grid[0] and mapped_x == True:
            x = round(v[0] - self.grid[0],3)
        elif v[0] <= 0 and mapped_x == True:
            x = round(v[0] + self.grid[0],3)
        else :
            x = round(v[0],3)

        if v[1] >= self.grid[1] and mapped_y == True:
            y = round(v[1] - self.grid[1],3)
        elif v[1] <= 0 and mapped_y == True:
            y = round(v[1] + self.grid[1],3)
        else :
            y = round(v[1],3)

        return np.array([x,y])
        
    def insert(self, point, score, real_object):
        def check(test,array):
            return any(np.array_equal([x[0], x[1]], test) for x in array)

        if self.checkIfInGrid(point) == True:
            if check(point, self.cache[int(point[0])][int(point[1])]) == True:
                #print("Point already present in cache : ", point)
                return [point, False]
            else:
                self.virtual_to_real[point[0]][point[1]] = real_object
                #self.real_to_virtual[real_object[0]][real_object[1]] = point

                point = np.append(point, score)
                self.cache[int(point[0])][int(point[1])].append(point) 

            return [point, True]
        else :
            new_point = self.findOriginalPoint(point, True, True)

            if check(new_point, self.cache[int(new_point[0])][int(new_point[1])]) == True:
                #print("Point already present in cache : ", point)
                return [point, False]
            else :
                self.virtual_to_real[new_point[0]][new_point[1]] = real_object
                #self.real_to_virtual[real_object[0]][real_object[1]] = new_point          

                new_point.append(old_point[-1])
                self.cache[int(new_point[0])][int(new_point[1])].append(new_point) 

            return [point, True]

    def delete(self, point, mapped_x, mapped_y, peak = False):

        if mapped_x == False and mapped_y == False:
            real_obj = self.virtual_to_real[point[0]][point[1]]

            ## Clear from the map
            if peak == False:
                try:
                    del self.virtual_to_real[point[0]][point[1]]
                    #del self.real_to_virtual[real_obj[0]][real_obj[1]]
                except :

                    print("Dictionary : ", self.real_to_virtual)
                    print("Real object : ", real_obj)
                    
            
            score = [x[2] for x in list(self.cache[int(point[0])][int(point[1])]) if x[0] == point[0] and x[1] == point[1]]

            if peak == False :
                self.cache[int(point[0])][int(point[1])] = [x for x in list(self.cache[int(point[0])][int(point[1])]) if x[0] != point[0] or x[1] != point[1]]

            return [real_obj, score, point] 
        else:
            new_point = self.findOriginalPoint(point, mapped_x, mapped_y)

            real_obj = self.virtual_to_real[new_point[0]][new_point[1]]

            if peak == False:
                try:

                    del self.virtual_to_real[new_point[0]][new_point[1]]
                    #del self.real_to_virtual[real_obj[0]][real_obj[1]]
                except:
                    print("Dictionary : ", self.real_to_virtual)
                    print("Real object : ", real_obj)
                    

            score = [x[2] for x in list(self.cache[int(new_point[0])][int(new_point[1])]) if x[0] == new_point[0] and x[1] == new_point[1]]

            if peak == False:
                self.cache[int(new_point[0])][int(new_point[1])] = [x for x in list(self.cache[int(new_point[0])][int(new_point[1])]) if x[0] != new_point[0] or x[1] != new_point[1]]

            return [real_obj, score, new_point] 


    def get_all_points(self):
        points = []
        g_a = 0
        g_b = 0
        for a in self.cache:
            for b in self.cache[a]:
                points.extend(self.cache[a][b])
                if a >= 10 and a <= 120 and b >= 10 and b <= 120:
                    g_a += len(self.cache[a][b])
                elif a >= 135 and a <= 305 and b >= 135 and b <= 305:
                    g_b += len(self.cache[a][b])
        return [points, g_a, g_b]

    def getRealObject(self, point):
        return self.virtual_to_real[point[0]][point[1]]

    def getVirtualObject(self, point):
        return self.real_to_virtual[point[0]][point[1]]

    
class CacheGridReal(CacheGrid):

    def __init__(self, capacity, dim, learning_rate, integral=False, grid_s=[313,313], policy="lfu"):
        CacheGrid.__init__(self, capacity, dim, learning_rate, integral=False, grid_s=[313,313])
        self.cache_size = 0
        self.freq_map = defaultdict(list)
        self.freq_sorted_list = []
        
    def initializeIterativeSearch(self, dim):
        self.obj_pos = objPosVirtual(self.grid)

    def insertInit(self, obj, iter, policy):
        if policy == "lfu" :
            [point, success] = self.obj_pos.insert(obj ,0, obj)
            if success == True:
                self.freq_map[0].append(obj)
                bisect.insort(self.freq_sorted_list, 0)
        elif policy == "lru" :
            [point, success] = self.obj_pos.insert(obj ,iter, obj)
            if success == True:
                self.freq_map[iter].append(obj)
                bisect.insort(self.freq_sorted_list, iter)

        ## Finish coding this up
        else :
            pass


    def updateVirtualObjectAndFreq(self, nearest_obj, new_obj_loc, mapped_x, mapped_y, iter, policy, points):        
        [real_obj, score, point_grid] = self.obj_pos.delete(nearest_obj, mapped_x, mapped_y, True)        

        score = score[0]        

        if policy == "lfu":
            score_new = score + 1
        elif policy == "lru" :
            score_new = iter
        ## Finish
        else:
            pass

            
        [obj_loc, success] = self.obj_pos.insert(new_obj_loc, score_new, real_obj)

        if success == True:        
            
            self.freq_map[score] = [x for x in self.freq_map[score] if (x[0] != point_grid[0] or x[1] != point_grid[1])]
            
            index = bisect.bisect(self.freq_sorted_list, score)
            
            self.freq_sorted_list.pop(index - 1)
            
            self.freq_map[score_new].append(obj_loc)
              
            bisect.insort(self.freq_sorted_list, score_new)              
              
            [real_obj, score, point_grid] = self.obj_pos.delete(nearest_obj, mapped_x, mapped_y)        


    def getReal(self, point):
        return self.obj_pos.getRealObject(point)
    
    def insert(self, obj, iter, policy):
        if policy == "lfu" :
            [point, success] = self.obj_pos.insert(obj, 0, obj)
        else :
            [point, success] = self.obj_pos.insert(obj, iter, obj)

        if success == True:
            self.evict()
            if policy == "lfu":
                self.freq_map[0].append(obj)
                bisect.insort(self.freq_sorted_list, 0)
            else :
                self.freq_map[iter].append(obj)
                bisect.insort(self.freq_sorted_list, iter)


    
    def updateRealObject(self, obj, iter, policy, virtual_object, orig_real_obj):
        self.obj_pos.virtual_to_real[virtual_object[0]][virtual_object[1]] = obj
#        print(self.obj_pos.real_to_virtual[orig_real_obj[0]][orig_real_obj[1]])
#        del self.obj_pos.real_to_virtual[orig_real_obj[0]][orig_real_obj[1]] 
#        self.obj_pos.real_to_virtual[orig_real_obj[0]][orig_real_obj[1]] = virtual_object


                
    def evict(self):
        least_freq = self.freq_sorted_list[0]

        obj_to_evict = self.freq_map[least_freq][0] 

        score = least_freq

        self.freq_map[score] = [x for x in self.freq_map[score] if x[0] != obj_to_evict[0] or x[1] != obj_to_evict[1]]

        if len(self.freq_map) == 0:
            del self.freq_map[score]

        self.obj_pos.delete(obj_to_evict, False, False)

        index = bisect.bisect(self.freq_sorted_list, least_freq)
        self.freq_sorted_list.pop(index - 1)


    def findNearestVirtual(self, vec):
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
            first = np.linalg.norm((c[:-1]-v), ord=1)
            if first > 4 * break_i:
                mapped_points = self.get_mapped_points(c)
                mapped = [(c, np.linalg.norm((c-v), ord=1)) for c in mapped_points]
                best = min(mapped, key=operator.itemgetter(1))                
                if best[0][0] == mapped_points[0][0] and best[0][1] == mapped_points[0][1]:
                    return [best[0], best[1], True, False]
                elif best[0][0] == mapped_points[1][0] and best[0][1] == mapped_points[1][1]:
                    return [best[0], best[1], False, True]
                else :
                    return [best[0], best[1], True, True]

            else:
                return [c , first, False, False]

        candidates = [dist(c, vec, break_i) for c in candidates]
        #print("len candidates : ", len(candidates), break_i)
        random.shuffle(candidates)
        best_candidate = min(candidates, key=operator.itemgetter(1))
        return [best_candidate[0], best_candidate[1], best_candidate[2], best_candidate[3]]



    def findNearestReal(self, vec):
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
                    candidates.extend([self.obj_pos.virtual_to_real[req_point[0]][req_point[1]] for req_point in self.obj_pos.cache[x][y1]])
                    candidates.extend([self.obj_pos.virtual_to_real[req_point[0]][req_point[1]] for req_point in self.obj_pos.cache[x][y2]])
                   
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
                    candidates.extend([self.obj_pos.virtual_to_real[req_point[0]][req_point[1]] for req_point in self.obj_pos.cache[x1][y]])
                    candidates.extend([self.obj_pos.virtual_to_real[req_point[0]][req_point[1]] for req_point in self.obj_pos.cache[x2][y]])            

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
                if best[0][0] == mapped_points[0][0] and best[0][1] == mapped_points[0][1]:
                    return [best[0], best[1], True, False]
                elif best[0][0] == mapped_points[1][0] and best[0][1] == mapped_points[1][1]:
                    return [best[0], best[1], False, True]
                else :
                    return [best[0], best[1], True, True]

            else:
                return [c , first, False, False]

        candidates = [dist(c, vec, break_i) for c in candidates]
        random.shuffle(candidates)
        best_candidate = min(candidates, key=operator.itemgetter(1))
        return [best_candidate[0], best_candidate[1], best_candidate[2], best_candidate[3]]


