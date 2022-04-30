import numpy as np
import multiprocessing as mp
from numpy.random import rand
import wolff

def calcEnergy(config, H = 0):
    energy = 0 
    N = np.shape(config)[0]
    for i in range(N):
        for j in range(N):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S + S*H
    return energy/(N*N)/2.

def calcMag(config):
    N = np.shape(config)[0]
    mag = np.sum(config)/(N*N)
    return abs(mag)

def standardDeviation(E_err, E):
    n = len(E_err)
    sigma_E = 0
    for i in range(n):
        sigma_E += (E_err[i] - E)**2
    return np.sqrt(sigma_E/n/(n-1))

def jackknife(E_err, E_err2, C, beta, N):
    n = len(E_err)
    C_jk = np.zeros(n)
    sigma_C = 0
    #calculating C_i - capacity without E_i
    for i in range(n):
        E1_arr = np.sum(np.delete(E_err, i))
        E2_arr = np.sum(np.delete(E_err2, i))
        C_jk[i] = (E2_arr/(n-1) - E1_arr/(n-1)*E1_arr/(n-1))*beta*N
    #calculating sum (C_i - C)^2
    for i in range(n):
        sigma_C += (C_jk[i] - C)**2
    return np.sqrt(sigma_C)

def wolffCalc(T, N, eqSteps, wolffSteps, i_border, j_border):
    config = wolff.initialstate(N, i_border, j_border)
    beta = 1/T
    beta2 = beta*beta
    prob = 1 - np.exp(-2*beta)
    #block for observables
    E1 = M1 = E2 = M2 = 0
    #block of errs
    E1_err, E2_err, M1_err, M2_err = np.zeros(wolffSteps), np.zeros(wolffSteps), np.zeros(wolffSteps), np.zeros(wolffSteps)
    for _ in range(eqSteps):
        wolff.wolffmove(config, prob, i_border, j_border)
    for i in range(wolffSteps):
        wolff.wolffmove(config, prob, i_border, j_border)
        Ene = calcEnergy(config)
        Mag = calcMag(config)

        #block for observables
        E1 += Ene
        E2 += Ene*Ene
        M1 += Mag
        M2 += Mag*Mag

        #block for errs
        E1_err[i] = Ene
        E2_err[i] = Ene*Ene
        M1_err[i] = Mag
        M2_err[i] = Mag*Mag

    #block for observables
    E = E1/wolffSteps
    C = (E2/wolffSteps - E1/wolffSteps*E1/wolffSteps)*beta2*N**2
    M = M1/wolffSteps
    X = (M2/wolffSteps - M1/wolffSteps*M1/wolffSteps)*beta*N**2

    #block for errs
    E_err = standardDeviation(E1_err, E)
    C_err = jackknife(E1_err, E2_err, C, beta2, N**2)
    M_err = standardDeviation(M1_err, M)
    X_err = jackknife(M1_err, M2_err, X, beta, N**2)

    return np.array([T, E, C, M, X, E_err, C_err, M_err, X_err])

N = 20
nt = 50
i_border = j_border = -1


eqSteps = 2**12
wolffSteps = 2**15

T = np.linspace(0.1, 5, nt)
#block for observables
E, C, M, X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)

pool_arg = [[T[i], N, eqSteps, wolffSteps, i_border, j_border] for i in range(nt)]

if __name__ == '__main__':
    pool = mp.Pool(processes=10) # Создание пула процессов
    results = np.array(pool.starmap(wolffCalc, pool_arg)) # Параллельный расчёт для всех температур

    f = open('critical_exponents\\wolff_data.txt', 'w')

    f.write('%s\n' %N)
    results = results.T
    for i in results:
        f.write('%s' %i[0])
        for tt in i[1:]:
            f.write(',%s' %tt)
        f.write('\n')

    f.close()