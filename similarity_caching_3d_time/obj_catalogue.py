
class ObjectCatalogueGrid3d:
    def __init__(self, dim_x, dim_y, dim_z, exp_type, filename):
        self.catalogue = []
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        #self.trace_f = open("traces/trace_" + filename + ".txt", "r")
        self.trace_f = open("new_traces/trace_" + filename + "_shuffle_norepeat_2.txt" , "r")
       
    def getRequest(self):
        req = self.trace_f.readline()
        req = req.strip().split(" ")        
        lat = int(req[3]) 
        long = int(req[4])

        return [[lat, long, int(req[1]), "y"]]

        reqs = []
        for lt in range(lat-2, lat+2):
            for ln in range(long-2, long+2):
                approx = "y"
                if lt < lat - 6 or lt > lat + 6:
                    approx = "y"
                if ln < long - 6 or ln > ln + 6:
                    approx = "y"
                reqs.append([lt%self.dim_x, ln%self.dim_y, int(req[1]), approx])
                
        return reqs
        
    def getNearest(self, point):
        r_point = [math.ceil(x) if x >= 0.5 else math.floor(x) for x in point]
        return r_point

