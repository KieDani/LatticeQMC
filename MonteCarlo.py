import numpy as np
import scipy as sp
import scipy.linalg as la
from multiprocessing import Pool
import multiprocessing
import functools
import configuration
import Hamiltonian as Ha

#not used yet
numberCores = 1
L=4
dim = 1
N = L**dim
T = 2.
kB = 1.38064852e-23
#not used yet!!!
#beta = 1/(T * kB)
stepsize = 25
deltaTau = T/stepsize
U = 2
t = 1
mu = U/2.
#Unterschied zwischen Vorlesung und Paper
v = np.arccosh(np.exp(U*deltaTau/2.))
lamb = v
#lamb = np.exp(U*deltaTau/2.)
C1 = 0.5*np.exp(-1*U*deltaTau/4.)
ha = Ha.Hamiltonian(L=L, U=U, mu=mu, t=t)
#conf is Object, config is array
#conf = configuration.Configuration(N=N, T=stepsize, seed=1234)


print('should be smaller than 0.1: ')
print(t*U*deltaTau**2)



#use the lecture way if determinants=False
def computeB_lsigma(l, sigma, config, determinants = True):
    if(determinants==True):
        K = ha.buildK()
        #check if there is a better way to calculate the matrix exponential
        tmp1 = la.expm(deltaTau*K)
        V_l = ha.buildV_l(l=l, config=config)
        #V_l is diagonal -> more effective way to calculate exp
        tmp2 = la.expm(sigma*v*V_l)
        B = np.dot(tmp1, tmp2)
        return B
    else:
        K = ha.buildK()
        # check if there is a better way to calculate the matrix exponential
        tmp1 = la.expm(deltaTau * K)
        V_l = ha.buildV_l(l=l, config=config)
        # V_l is diagonal -> more effective way to calculate exp
        tmp2 = la.expm(sigma * lamb * V_l)
        B = np.dot(tmp1, tmp2)
        return B



#use the lecture way if determinants=False
def computeM_sigma(sigma, config, determinants = True):
    M = np.zeros((N,N), dtype=np.float64)
    for i in range(1,N):
        M[i,i] = 1.

    lmax = stepsize - 1
    Bs = computeB_lsigma(l=lmax, sigma=sigma, config=config, determinants=determinants)
    lmax -= 1
    while (lmax >= 0):
        B = computeB_lsigma(l=lmax, sigma=sigma, config=config, determinants=determinants)
        Bs = np.dot(Bs, B)
        lmax -= 1
    M = M + Bs
    return M



def computeG_sigma(sigma, M_sigma):
    G = np.linalg.inv(M_sigma)
    print('G_'+ str(sigma))
    #print(G)
    return G


def warmup(sweeps=int(0.5*N*stepsize), seed=1234, determinants=True):
    if(determinants==True):
        print('aktueller Prozess:')
        print(multiprocessing.current_process().pid)
        seed *= multiprocessing.current_process().pid
        #global conf
        conf = configuration.Configuration(N=N, T=stepsize, seed=seed)
        config = conf.get()
        configOld = np.copy(config)
        old = np.linalg.det(computeM_sigma(sigma=+1, config=config)) * np.linalg.det(computeM_sigma(sigma=-1, config=config))
        for a in range(0, sweeps):
            for i in range(0,N):
                for l in range(0,stepsize):
                    print('warmup step ' + str(a))
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
                       configOld = np.copy(config)
                    else:
                        print('do not accept move :(')
                        #restore old state again
                        conf.update(i, l)
                        config = conf.get()

                    print('_______________________________________')
                    #print(config)
                    #print('_______________________________________')
        return conf
    else:
        #Greensfunction does not need to be calculated anew every step!
        print('aktueller Prozess:')
        print(multiprocessing.current_process().pid)
        seed *= multiprocessing.current_process().pid
        # global conf
        conf = configuration.Configuration(N=N, T=stepsize, seed=seed)
        config = conf.get()
        G_up = np.linalg.inv(computeM_sigma(sigma=+1, config=config, determinants=determinants))
        G_down = np.linalg.inv(computeM_sigma(sigma=+1, config=config, determinants=determinants))
        for a in range(0, sweeps):
            for i in range(0,N):
                for l in range(0,stepsize):
                    config = conf.get()
                    print('warmup step ' + str(a))
                    # Random number between 0 and 1
                    r = np.random.rand()
                    print('Random number ' + str(r))
                    d_up = 1+(1-np.linalg.inv(computeM_sigma(sigma=+1, config=config, determinants=determinants))[i][i])*(np.exp(-2*lamb*config[i][l])-1)
                    d_down = 1+(1-np.linalg.inv(computeM_sigma(sigma=-1, config=config, determinants=determinants))[i][i])*(np.exp(2*lamb*config[i][l])-1)
                    Prob = d_up + d_down
                    print('Probability ' + str(Prob))
                    if (r < Prob):
                        print('accept move')
                        c_up = np.zeros(N, dtype=np.float64)
                        c_up[i] = np.exp(-2 * lamb * config[i][l]) - 1
                        c_down = np.zeros(N, dtype=np.float64)
                        c_down[i] = np.exp(+2 * lamb * config[i][l]) - 1
                        c_up = -1 * (np.exp(-2 * lamb * config[i][l]) - 1) * G_up[i, :] + c_up
                        c_down = -1 * (np.exp(2 * lamb * config[i][l]) - 1) * G_down[i, :] + c_down
                        b_up = G_up[:, i] / (1. + c_up[i])
                        b_down = G_down[:, i] / (1. + c_down[i])
                        G_up = G_up - np.outer(b_up, c_up)
                        G_down = G_down - np.outer(b_down, c_down)

                        conf.update(n=i, t=l)
                    else:
                        print('do not accept move :(')
        return conf








def measureG(sweeps, thermalization=int(0.5*N*stepsize), seed=1234, determinants=True):
    def delta(i, j):
        if(i==j): return 1
        else: return 0

    if(determinants==True):
        number = 0
        G_up = 0
        G_down = 0
        conf = warmup(sweeps=thermalization, seed=seed)
        config = conf.get()
        old = np.linalg.det(computeM_sigma(sigma=+1, config=config, determinants=determinants)) * np.linalg.det(
            computeM_sigma(sigma=-1, config=config, determinants=determinants))
        M_up = computeM_sigma(sigma=+1, config=config, determinants=determinants)
        M_down = computeM_sigma(sigma=-1, config=config, determinants=determinants)
        G_up_tmp = np.linalg.inv(M_up)
        G_down_tmp = np.linalg.inv(M_down)
        for a in range(0, sweeps):
            for i in range(0,N):
                for l in range(0,stepsize):
                    print('Step ' + str(a))
                    conf.update(i, l)
                    config = conf.get()
                    M_up = computeM_sigma(sigma=+1, config=config, determinants=determinants)
                    M_down = computeM_sigma(sigma=-1, config=config, determinants=determinants)
                    new = np.linalg.det(M_up) * np.linalg.det(M_down)
                    # Random number between 0 and 1
                    r = np.random.rand()
                    print('Random number ' + str(r))
                    Prob = new / old
                    print('Probability ' + str(Prob))
                    if (r < Prob):
                        print('accept move')
                        old = new
                        G_up_tmp = np.linalg.inv(M_up)
                        print('Greensfunction up')
                        print(G_up_tmp)
                        G_down_tmp = np.linalg.inv(M_down)
                        print('Greensfunction down')
                        print(G_down_tmp)
                    else:
                        print('do not accept move :(')
                        # restore old state again
                        conf.update(i, l)
                        config = conf.get()
                    G_up += G_up_tmp
                    G_down += G_down_tmp
                    number += 1

                    print('_______________________________________')
        G_up = G_up/number
        G_down = G_down/number
        print('hallo')
        return G_up, G_down
    else:
        number = 0
        # conf = configuration.Configuration(N=N, T=stepsize, seed=seed)
        conf = warmup(sweeps=thermalization, seed=seed)
        config = conf.get()
        G_up2 = 0
        G_down2 = 0
        G_up = np.linalg.inv(computeM_sigma(sigma=+1, config=config, determinants=determinants))
        G_down = np.linalg.inv(computeM_sigma(sigma=+1, config=config, determinants=determinants))
        for a in range(0, sweeps):
            for i in range(0,N):
                for l in range(0,stepsize):
                    print('Step ' + str(a))
                    config = conf.get()
                    # Random number between 0 and 1
                    r = np.random.rand()
                    print('Random number ' + str(r))
                    d_up = 1 + (1 - np.linalg.inv(computeM_sigma(sigma=+1, config=config, determinants=determinants))[i][i]) * (
                                np.exp(-2 * lamb * config[i][l]) - 1)
                    d_down = 1 + (
                                1 - np.linalg.inv(computeM_sigma(sigma=-1, config=config, determinants=determinants))[i][i]) * (
                                         np.exp(2 * lamb * config[i][l]) - 1)
                    Prob = d_up + d_down
                    print('Probability ' + str(Prob))
                    if (r < Prob):
                        print('accept move')
                        c_up = np.zeros(N, dtype=np.float64)
                        c_up[i] = np.exp(-2*lamb*config[i][l])-1
                        c_down = np.zeros(N, dtype=np.float64)
                        c_down[i] = np.exp(+2 * lamb * config[i][l]) - 1
                        c_up = -1*(np.exp(-2*lamb*config[i][l])-1) * G_up[i,:] + c_up
                        c_down = -1 * (np.exp(2 * lamb * config[i][l]) - 1) * G_down[i, :] + c_down
                        b_up = np.zeros(N, dtype=np.float64)
                        b_up = G_up[:,i]/(1. + c_up[i])
                        b_down = np.zeros(N, dtype=np.float64)
                        b_down = G_down[:, i] / (1. + c_down[i])
                        G_up = G_up - np.outer(b_up, c_up)
                        G_down = G_down - np.outer(b_down, c_down)

                        conf.update(n=i, t=l)
                    else:
                        print('do not accept move :(')
                    G_up2 += G_up
                    G_down2 += G_down
                    number += 1

            print('_______________________________________')
        G_up2 = G_up2 / number
        G_down2 = G_down2 / number
        print('hallo')
        return G_up2, G_down2




#number of sweeps is not exact because of integer division
def measure(thermalization, sweeps, determinants=True):
    global tmp
    G_up = 0
    G_down = 0
    with Pool(numberCores) as p:
        sweep = np.ones(numberCores, dtype=np.int64) * int(round(float(sweeps)/numberCores))
        result = p.map(functools.partial(measureG, thermalization=thermalization, seed=1234, determinants=determinants), sweep)
        print(result)
        for i in range(0, numberCores):
            tmp = result[i]
            G_up += tmp[0]
            G_down += tmp[1]
        G_up = G_up/numberCores
        G_down = G_down/numberCores
    print('saving')
    np.savetxt('G_up_N' + str(N) + 'U' + str(U) + 't' + str(t) + 'mu' + str(mu) + 'T' + str(T) + 'step' + str(stepsize) + 'det' + str(determinants) + '.txt', G_up)
    np.savetxt('G_down_N' + str(N) + 'U' + str(U) + 't' + str(t) + 'mu' + str(mu) + 'T' + str(T) + 'step' + str(stepsize) + 'det' + str(determinants) + '.txt', G_down)
    return G_up, G_down



#DOS in real space
def calculateDOS_i_sigma(G_sigma):
    DOS_sigma = list()
    for i in range(0,N):
        DOS_sigma.append(1-G_sigma[i,i])
    return DOS_sigma


#not working at the moment!
#only working for simple chain. Use definition of lattice in future
def DFT(k, DOS_sigma):
    #lattice constant
    #a = 5e-10
    a=1
    DOS_k = 0
    for i in range(0,N):
        DOS_k += np.exp(complex(0,1)*a*i*k)*DOS_sigma[i]
    DOS_k *= 1./N
    return DOS_k






G_up, G_down = measure(thermalization=500, sweeps=3200, determinants=True)
G_up, G_down = measure(thermalization=500, sweeps=3200, determinants=False)
#np.savetxt('G_up.txt', G_up)
#np.savetxt('G_down.txt', G_down)

# G_up = np.loadtxt('G_up_U2t1mu2T4.0step25detFalse.txt')
# G_down = np.loadtxt('G_down_U2t1mu2T4.0step25detFalse.txt')
# print(G_up)
# print(G_down)
# DOS_up = calculateDOS_i_sigma(G_up)
# print('DOS up')
# print(DOS_up)
# print(np.sum(DOS_up))
# DOS_down = calculateDOS_i_sigma(G_down)
# print('DOS down')
# print(DOS_down)
# print(np.sum(DOS_down))
