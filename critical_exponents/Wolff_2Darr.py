import numpy as np
from numpy.random import rand
from mpi4py import MPI
import wolff

'''
Эта программа нужна для демонстрации
алгоритма Вольфа в широком диапазоне
температур.
'''

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
    #вычисление C_i - теплоёмкость без E_i
    for i in range(n):
        E1_arr = np.sum(np.delete(E_err, i))
        E2_arr = np.sum(np.delete(E_err2, i))
        C_jk[i] = (E2_arr/(n-1) - E1_arr/(n-1)*E1_arr/(n-1))*beta*N
    #вычисление суммы (C_i - C)^2
    for i in range(n):
        sigma_C += (C_jk[i] - C)**2
    return np.sqrt(sigma_C)

def wolffcalc(T, N, eqSteps, wolffSteps, i_border, j_border):
    config = wolff.initialstate(N, i_border, j_border)
    beta = 1/T
    beta2 = beta*beta
    prob = 1 - np.exp(-2*beta)

    #наблюдаемые
    E1 = M1 = E2 = M2 = 0

    #погрешности
    E1_err, E2_err, M1_err, M2_err = np.zeros(wolffSteps), np.zeros(wolffSteps), np.zeros(wolffSteps), np.zeros(wolffSteps)

    for _ in range(eqSteps):
        wolff.wolffmove(config, prob, i_border, j_border)
    for i in range(wolffSteps):
        wolff.wolffmove(config, prob, i_border, j_border)
        Ene = calcEnergy(config)
        Mag = calcMag(config)

        #наблюдаемые
        E1 += Ene
        E2 += Ene*Ene
        M1 += Mag
        M2 += Mag*Mag

        #погрешности
        E1_err[i] = Ene
        E2_err[i] = Ene*Ene
        M1_err[i] = Mag
        M2_err[i] = Mag*Mag

    #наблюдаемые
    E = E1/wolffSteps
    C = (E2/wolffSteps - E1/wolffSteps*E1/wolffSteps)*beta2*N**2
    M = M1/wolffSteps
    X = (M2/wolffSteps - M1/wolffSteps*M1/wolffSteps)*beta*N**2

    #погрешности
    E_err = standardDeviation(E1_err, E)
    C_err = jackknife(E1_err, E2_err, C, beta2, N**2)
    M_err = standardDeviation(M1_err, M)
    X_err = jackknife(M1_err, M2_err, X, beta, N**2)

    return np.array([T, E, C, M, X, E_err, C_err, M_err, X_err])

# Граничные условия:
# 1 - фиксированные спины s = (+1)
# 0 - непереодические (на квадрате)
# -1 - переодические (на торе)

# Объявление объектов MPI:
world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()

# Условия моделирования

nt = 50 # число значений температуры
N = 64 # число спинов на стороне решётки, т.е. решётка размера N*N
eqSteps = 2**12
wolffSteps = 2**14
i_border, j_border = 0, 0
T = np.linspace(0.01, 5, nt) # Массив температур

# составление списка аргументов для функции wolffcalc
pool_args = [[T[i], N, eqSteps, wolffSteps, i_border, j_border] for i in range(nt)]

# данный код раздаёт части списка pool_arg разным процессам
workloads = [nt // world_size for _ in range(world_size)]
for i in range(nt%world_size):
    workloads[i] += 1
my_start = 0
for i in range(my_rank):
    my_start += workloads[i]
my_end = my_start + workloads[my_rank]

if my_rank == 0:
    start_time = MPI.Wtime()

    # Рассчёт своих данных
    results = wolffcalc(*pool_args[my_start])
    output_size = len(results)

    for i in range(my_start + 1, my_end):
        results = np.append(results, wolffcalc(*pool_args[i]))

    # Получение данных от других процессов
    for i in range(1, world_size):
        result = np.empty(workloads[i]*output_size)
        world_comm.Recv([result, MPI.DOUBLE], source=i, tag=1)
        results = np.append(results, result)
    results = results.reshape((nt,output_size)).T # матрица, в строках которой данные конкретной величины для каждой температуры

    # Запись данных в файл для дальнейшей обработки
    f = open('critical_exponents\\wolff.txt', 'w')

    f.write('%s\n' %N)
    for row in results:
        output = ''
        first = True
        for elem in row:
            output += ('' if first else '; ') + str(elem)
            first = False
        f.write(output + '\n')

    f.close()

    end_time = MPI.Wtime()
    print('Calculation time: ' + str(end_time - start_time))
else:
    result = wolffcalc(*pool_args[my_start])
    for i in range(my_start + 1, my_end):
        result = np.append(result, wolffcalc(*pool_args[i]))
    world_comm.Send([result, MPI.DOUBLE], dest=0, tag=1)