import numpy as np

#builds Hubbard matrix: H=-t*sum(c_i*c_j)+(U/2-mu)*sum(c_i*c_i)+U*sum((n_i,up - 1/2)*(n_i,down-1/2))
class Hamiltonian:

    def __init__(self, L, U, t, mu, dim=1):
        self.L = L
        self.dim = dim
        self.U = U
        self.t = t
        self.mu = mu
        #graph[0] is edges, graph[1] is sites
        self.graph = self.buildGraph(L=L, dim=dim)
        self.K = self.buildK()

    #builds hyperqubic graph. Only 1D at the moment
    #L...sites of one size -> N=L**dim; dim...dimension of the hyperqubic lattice
    def buildGraph(self, L, dim=1):
        #dim=1
        edges = list()
        sites = list()
        for i in range(0, L-1):
            edges.append([i,i+1])
            sites.append(i)
        edges.append([L-1,0])
        sites.append(L-1)
        #print(edges)
        #print(sites)
        graph = (edges, sites)
        return graph


    #could be implemented to use with exact diagonalization
    def buildHubbard(self):
        pass

    #only for 1D-lattice at the moment
    #k_ij= -t(del_i,j+1 + del_i,j-1) + (U/2 - mu)del_i,j
    def buildK(self):
        graph = self.graph
        edges = graph[0]
        sites = graph[1]
        N = self.L**self.dim
        U = self.U
        mu = self.mu
        t = self.t
        K = np.zeros((N,N), dtype=np.float64)
        #hopping
        for i in range(0,len(edges)):
            tmp  = edges[i]
            K[tmp[0], tmp[1]] = t
            K[tmp[1], tmp[0]] = t
        #on-site interaction
        for i in range(0,len(sites)):
            tmp  = sites[i]
            K[tmp, tmp] = U/2 - mu
            K[tmp, tmp] = U / 2 - mu
        #print('K = ')
        #print(K)
        return K


    #l is the aktual time slice; conf is the configuration of the H-S-Spins (array, not Object)
    def buildV_l(self, l, config):
        N = self.L**self.dim
        V_l = np.zeros((N,N), dtype=np.int8)
        #Spins at time slice l
        h = config[:,l]
        for i in range(0,N):
            V_l[i,i] = h[i]
        #print('V_' + str(l) + ' = ')
        #print(V_l)
        return V_l