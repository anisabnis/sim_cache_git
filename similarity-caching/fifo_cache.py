import numpy as np

class FIFO:
    def __init__(self, size):
        self.cache = []
        self.size = size

    def insert(self, obj):
        self.cache.insert(0, [obj[0], obj[1], 0])
        if len(self.cache) > self.size:
            return self.remove()
        return None

    def remove(self):
        last_ele = self.cache[-1]
        self.cache = self.cache[:-1]
        return last_ele

    def findNearest(self, p, t):
        closest_dst = t
        closest_point = 0
        i = 0
        for c in self.cache:
            c = np.array([c[0], c[1]])
            d = np.linalg.norm(p - c, ord=1)
            if d < closest_dst:
                closest_dst = d
                closest_point = i
            i += 1

        return [closest_point, closest_dst]
            
    def updateScore(self, req_obj, threshold):
        res = self.findNearest(req_obj, threshold)
        if res[1] < threshold:
            self.cache[res[0]][2] += threshold - res[1]

    
