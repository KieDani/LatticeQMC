import numpy as np
import scipy as sp
import scipy.linalg as la
from multiprocessing import Pool
import multiprocessing
import functools
import configuration
import Hamiltonian as Ha

#not used yet
numberCores = 4
L=5
dim = 1
N = L**dim
T = 25.
kB = 1.38064852e-23
beta = 1/(T * kB)
stepsize = 200
deltaTau = T/stepsize
U = 2
t = 1
mu = 2
#Unterschied zwischen Vorlesung und Paper
v = np.arccosh(np.exp(U*deltaTau/2.))
#lamb = v
lamb = np.exp(U*deltaTau/2.)
C1 = 0.5*np.exp(-1*U*deltaTau/4.)
ha = Ha.Hamiltonian(L=L, U=U, mu=mu, t=t)
#conf is Object, config is array
#conf = configuration.Configuration(N=N, T=stepsize, seed=1234)





#I don't know if I have to use dot or *
def mult(A,B, dot=True):
    if(dot==True):
        return np.dot(A,B)
    else:
        return A*B


def MCsweep(l, t, L, conf):
    conf.update(n=l, t=t)




#use the lecture way if determinants=False
def computeB_lsigma(l, sigma, config, determinants = True):
    if(determinants==True):
        K = ha.buildK()
        #check if there is a better way to calculate the matrix exponential
        tmp1 = la.expm(deltaTau*K)
        V_l = ha.buildV_l(l=l, config=config)
        #V_l is diagonal -> more effective way to calculate exp
        tmp2 = la.expm(sigma*v*V_l)
        B = mult(tmp1, tmp2)
        return B
    else:
        K = ha.buildK()
        # check if there is a better way to calculate the matrix exponential
        tmp1 = la.expm(deltaTau * K)
        V_l = ha.buildV_l(l=l, config=config)
        # V_l is diagonal -> more effective way to calculate exp
        tmp2 = la.expm(sigma * lamb * V_l)
        B = mult(tmp1, tmp2)
        return B



#use the lecture way if determinants=False
def computeM_sigma(sigma, config, determinants = True):
    M = np.zeros((N,N), dtype=np.float64)
    for i in range(1,N):
        M[i,i] = 1.

    if(determinants==True):
        lmax = stepsize-1
        Bs = computeB_lsigma(l=lmax, sigma=sigma, config=config, determinants=determinants)
        lmax -= 1
        while(lmax >= 0):
            B = computeB_lsigma(l=lmax, sigma=sigma, config=config, determinants=determinants)
            Bs = mult(Bs, B)
            lmax -= 1
        M = M + Bs
        return M
    else:
        lmax = 0
        Bs = computeB_lsigma(l=lmax, sigma=sigma, config=config, determinants=determinants)
        lmax += 1
        while (lmax < stepsize):
            B = computeB_lsigma(l=lmax, sigma=sigma, config=config, determinants=determinants)
            Bs = mult(Bs, B)
            lmax += 1
        M = M + Bs
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
        for i in range(0, sweeps):
            print('warmup step ' + str(i))
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
               #display only for small matrices
               if(N*stepsize<=100):
                    print(configOld-config)
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
        for i in range(0, sweeps):
            config = conf.get()
            print('warmup step ' + str(i))
            i = np.random.randint(0,N)
            l = np.random.randint(0, stepsize)
            # Random number between 0 and 1
            r = np.random.rand()
            print('Random number ' + str(r))
            d_up = 1+(1-np.linalg.inv(computeM_sigma(sigma=+1, config=config, determinants=determinants))[i][i])*(np.exp(-2*lamb*config[i][l])-1)
            d_down = 1+(1-np.linalg.inv(computeM_sigma(sigma=-1, config=config, determinants=determinants))[i][i])*(np.exp(2*lamb*config[i][l])-1)
            Prob = d_up + d_down
            print('Probability ' + str(Prob))
            if(r<Prob):
                print('accept move')
                conf.update(n=i, t=l)
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
        for i in range(0, sweeps):
            print('Step ' + str(i))
            G_up_tmp = 0
            G_down_tmp = 0
            i = np.random.randint(0, N)
            l = np.random.randint(0, stepsize)
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
        for i in range(0, sweeps):
            print('Step ' + str(i))
            config = conf.get()
            i = np.random.randint(0, N)
            l = np.random.randint(0, stepsize)
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
    np.savetxt('G_up_U' + str(U) + 't' + str(t) + 'mu' + str(mu) + 'T' + str(T) + 'step' + str(stepsize) + 'det' + str(determinants) + '.txt', G_up)
    np.savetxt('G_down_U' + str(U) + 't' + str(t) + 'mu' + str(mu) + 'T' + str(T) + 'step' + str(stepsize) + 'det' + str(determinants) + '.txt', G_down)
    return G_up, G_down



#DOS in real space
def calculateDOS_i_sigma(G_sigma):
    DOS_sigma = list()
    for i in range(0,N):
        DOS_sigma.append(1-G_sigma[i,i])
    return DOS_sigma


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






G_up, G_down = measure(thermalization=500, sweeps=2000, determinants=False)
#np.savetxt('G_up.txt', G_up)
#np.savetxt('G_down.txt', G_down)

#G_up = np.loadtxt('G_up.txt')
#G_down = np.loadtxt('G_down.txt')
#
# print(G_up)
# print(G_down)
#
#DOS_up = calculateDOS_i_sigma(G_up)
#print('DOS up')
#print(DOS_up)
#DOS_down = calculateDOS_i_sigma(G_down)
#print('DOS down')
#print(DOS_down)
#
# k = np.linspace(0, np.pi, 100)
# DOS_k = DFT(k, DOS_up)
# print('DOS')
# print(DOS_k)



# conf = configuration.Configuration(N=N, T=stepsize, seed=1234)
# config=conf.get()
# M_det = computeM_sigma(sigma=+1, config=config, determinants=True)
# M_int = computeM_sigma(sigma=+1, config=config, determinants=False)
# print(M_det)
# print('')
# print(M_int)