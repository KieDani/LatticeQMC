import numpy as np
import scipy as sp
import scipy.linalg as la
import configuration
import Hamiltonian as Ha


L=5
T = 100.
#no k_B used yet
beta = 1/T
stepsize = 100
deltaTau = T/stepsize
U = 2
t = 1
mu = 2
v = np.arccosh(np.exp(U*deltaTau/2.))
C1 = 0.5*np.exp(-1*U*deltaTau/4.)
ha = Ha.Hamiltonian(L=L, U=U, mu=mu, t=t)
#conf is Object, config is array
conf = configuration.Configuration(N=L, T=stepsize)


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
    B = np.dot(tmp1, tmp2)
    #print(B)
    return B





config = conf.get()
computeB_lsigma(2, 1, config)