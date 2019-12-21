import numpy as np
import scipy as sp
import scipy.linalg as la
import configuration
import Hamiltonian as Ha


L=3
dim = 1
N = L**dim
T = 100.
#no k_B used yet
beta = 1/T
stepsize = 20
deltaTau = T/stepsize
U = 2
t = 1
mu = 2
v = np.arccosh(np.exp(U*deltaTau/2.))
C1 = 0.5*np.exp(-1*U*deltaTau/4.)
ha = Ha.Hamiltonian(L=L, U=U, mu=mu, t=t)
#conf is Object, config is array
conf = configuration.Configuration(N=N, T=stepsize)


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
    return B



def computeM_sigma(sigma, config):
    M = np.zeros((N,N), dtype=np.float64)
    for i in range(1,N):
        M[i,i] = 1.
    Bs = computeB_lsigma(l=0, sigma=sigma, config=config)
    for l in range(1, stepsize):
        B = computeB_lsigma(l=l, sigma=sigma, config=config)
        #don't know if I have to use dot or elementwise *
        Bs = mult(B, Bs)
    M = M + Bs
    print(np.linalg.det(M))
    print(M)
    return M







#config = conf.get()
#computeB_lsigma(2, 1, config)
#computeM_sigma(1,config)



config = conf.get()
x = computeM_sigma(sigma=1, config=config)
#a = np.linalg.det(computeM_sigma(sigma=+1, config=config))*np.linalg.det(computeM_sigma(sigma=-1, config=config))
#print(np.linalg.det(computeM_sigma(sigma=-1, config=config)))
conf.update(2,7)
config = conf.get()
#b = np.linalg.det(computeM_sigma(sigma=+1, config=config))*np.linalg.det(computeM_sigma(sigma=-1, config=config))
y= computeM_sigma(sigma=1, config = config)
#print(b/a)
print(x-y)