import numpy as np

class FIFO:
    def __init__(self, size):
        self.cache = []
        self.size = size

    def insert(self, obj):
        self.cache.insert(0, [obj[0], obj[1], obj[2], 0])
        if len(self.cache) > self.size:
            return self.remove()
        return None

    def remove(self):
        last_ele = self.cache[-1]
        self.cache = self.cache[:-1]
        return last_ele

    def findNearest(self, p, t):
        closest_dst = 10000
        closest_point = 0
        i = 0
        for c in self.cache:
            c = np.array([c[0], c[1], c[2]])
            d = np.linalg.norm(p[:2] - c[:2], ord=1) + (p[2] - c[2])
            if d < closest_dst:
                closest_dst = d
                closest_point = i
            i += 1

        return [closest_point, closest_dst]
            
    def updateScore(self, req_obj, threshold):
        res = self.findNearest(req_obj, threshold)
        if res[1] < threshold:
            self.cache[res[0]][3] += threshold - res[1]

    
