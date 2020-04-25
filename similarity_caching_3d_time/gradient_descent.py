import numpy as np

class StochasticGradientDescent:
    def __init__(self, learning_rate, grid):
        self.alpha = learning_rate
        self.grid = grid

    def descent(self, nearest_object, current_object, type="gaussian", policy="lfu"):

        def derivative_l2(nearest_object, current_object):
            return 2 * (nearest_object - current_object)

        def derivative_l1(nearest_object, current_object):            
            #return np.array([-1 if nearest_object[i] - current_object[i] < 0 else 1 for i in range(len(nearest_object))])
            d = []
            for i in range(len(nearest_object)):
                if nearest_object[i] - current_object[i] < 0:
                    d.append(-1)
                elif nearest_object[i] - current_object[i] == 0:
                    d.append(0)
                else :
                    d.append(1)

            return d

        if type == "gaussian_real":
            if policy == "dual" or policy == "fifo":
                nearest_object = nearest_object[:-2]
            else :
                nearest_object = nearest_object[:-1]

        d = derivative_l1(nearest_object, current_object)
        #n = nearest_object - self.alpha * d
        n = np.array([nearest_object[0] - self.alpha * d[0], nearest_object[1] - self.alpha * d[1], nearest_object[2] - self.alpha * d[2]])

        try:
            n[0] = n[0]%self.grid[0]
            n[1] = n[1]%self.grid[1]
            n[2] = n[2]%self.grid[2]
            n = [round(x,3) for x in n]
            return n

        except:
            print("Error in descent : ", n, nearest_object)
