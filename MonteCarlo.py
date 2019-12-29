import numpy as np
import scipy as sp
import scipy.linalg as la
import configuration
import Hamiltonian as Ha


L=7
dim = 1
N = L**dim
T = 5.
#no k_B used yet
beta = 1/T
stepsize = 200
deltaTau = T/stepsize
U = 5
t = 1
mu = 2
#Unterschied zwischen Vorlesung und Paper
v = np.arccosh(np.exp(U*deltaTau/2.))
#v = np.exp(U*deltaTau/2.)
C1 = 0.5*np.exp(-1*U*deltaTau/4.)
ha = Ha.Hamiltonian(L=L, U=U, mu=mu, t=t)
#conf is Object, config is array
conf = configuration.Configuration(N=N, T=stepsize, seed=1234)


#I don't know if I have to use dot or *
def mult(A,B, dot=True):
    if(dot==True):
        return np.dot(A,B)
    else:
        return A*B


def MCsweep(l, t, L, conf):
    conf.update(n=l, t=t)





def computeB_lsigma(l, sigma, config):
    K = ha.buildK()
    tmp1 = la.expm(deltaTau*K)
    #print(K)
    #print(tmp1)
    V_l = ha.buildV_l(l=l, config=config)
    #print(V_l)
    #V_l is diagonal -> more effective way to calculate exp
    tmp2 = la.expm(sigma*v*V_l)
    #print(tmp2)
    #don't know, if I have to use np.dot or elementwise *
    B = mult(tmp1, tmp2)
    #print(B)
    #print('B_'+str(l)+str(sigma))
    #print(B)
    return B



def computeM_sigma(sigma, config):
    M = np.zeros((N,N), dtype=np.float64)
    for i in range(1,N):
        M[i,i] = 1.
    # Bs = computeB_lsigma(l=0, sigma=sigma, config=config)
    # for l in range(1, stepsize):
    #     B = computeB_lsigma(l=l, sigma=sigma, config=config)
    #     #don't know if I have to use dot or elementwise *
    #     Bs = mult(B, Bs)
    lmax = stepsize-1
    Bs = computeB_lsigma(l=lmax, sigma=sigma, config=config)
    lmax -= 1
    while(lmax >= 0):
        B = computeB_lsigma(l=lmax, sigma=sigma, config=config)
        Bs = mult(Bs, B)
        lmax -= 1
    M = M + Bs
    #print('Determinante ' + str(sigma))
    #print(np.linalg.det(M))
    #print(M)
    return M




def computeG_sigma(sigma, M_sigma):
    G = np.linalg.inv(M_sigma)
    print('G_'+ str(sigma))
    #print(G)
    return G

#Probability of acceptance of a spinfilp at site i and time l
def computeProbability(i, l, G_up, G_down, config):
    #look at definition of v again
    d_up = 1+(1-G_up[i,i])*(np.exp(-2*v*config[i,l])-1)
    d_down = 1+(1-G_down[i,i])*(np.exp(2*v*config[i,l])-1)
    return d_up+d_down





#config = conf.get()
#computeB_lsigma(2, 1, config)
#computeM_sigma(1,config)


def computeProbability_determinants():
    conf = configuration.Configuration(N=N, T=stepsize, seed=1234)
    config = conf.get()
    #x = computeM_sigma(sigma=1, config=config)
    a = np.linalg.det(computeM_sigma(sigma=+1, config=config))*np.linalg.det(computeM_sigma(sigma=-1, config=config))
    #print(np.linalg.det(computeM_sigma(sigma=-1, config=config)))
    conf.update(1,34)
    config = conf.get()
    b = np.linalg.det(computeM_sigma(sigma=+1, config=config))*np.linalg.det(computeM_sigma(sigma=-1, config=config))
    y= computeM_sigma(sigma=1, config = config)
    #print('Ergebnis')
    print(b/a)
    print(x-y)


#computeProbability_determinants()

def computeProbability_intelligent():
    conf = configuration.Configuration(N=N, T=stepsize, seed=1234)
    config = conf.get()
    M_up = computeM_sigma(sigma=+1, config=config)
    M_down = computeM_sigma(sigma=-1, config=config)
    G_up = computeG_sigma(sigma=+1,M_sigma=M_up)
    G_down = computeG_sigma(sigma=-1, M_sigma=M_down)
    i = 1
    l = 34
    p = computeProbability(i=i, l=l, G_up=G_up, G_down=G_down, config=config)
    print('Probability = ' + str(p))


#computeProbability_intelligent()

def warmup(sweeps=300):
    conf = configuration.Configuration(N=N, T=stepsize, seed=12345)
    config = conf.get()
    old = np.linalg.det(computeM_sigma(sigma=+1, config=config)) * np.linalg.det(computeM_sigma(sigma=-1, config=config))
    for i in range(0, sweeps):
        i = np.random.randint(0,N)
        l = np.random.randint(0, stepsize)
        conf.update(i,l)
        config = conf.get()
        new = np.linalg.det(computeM_sigma(sigma=+1, config=config)) * np.linalg.det(computeM_sigma(sigma=-1, config=config))
        #Random number between 0 and 1
        r = np.random.rand()
        print('Random number ' + str(r))
        Prob = new/old
        print('Probability ' + str(Prob))
        if(r<Prob):
           print('accept move')
           old = new
        else:
            print('do not accept move :(')
            #restore old state again
            conf.update(i, l)
            config = conf.get()



warmup()