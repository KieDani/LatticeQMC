import numpy as np


class Configuration:




    #N...number lattice sites; T...number time slices (per site)
    def __init__(self, N, T, seed=123):
        np.random.seed(seed)
        self.N = N
        self.T = T
        self.config = np.zeros((N,T), dtype=np.int8)
        self.initialize()



    #random initialization of the array config
    def initialize(self):
        def pmone():
            tmp = np.random.rand()
            if(tmp < 0.5):
                return -1
            else:
                return 1

        for n in range(0, self.N):
            for t in range(0,self.T):
                self.config[n,t] = pmone()

    #spinflip at site n and time t
    def update(self, n, t):
        self.config[n,t] *= -1


    #returns the array config
    def get(self):
        return self.config


    #returns an element of the array config
    def get_index(self, n, t):
        return self.config[n,t]


