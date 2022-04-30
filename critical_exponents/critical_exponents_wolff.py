import numpy as np
import wolff
from numpy.random import rand
from mpi4py import MPI

# Вычисление намагниченности решётки на спин
def calcMag(config):
    N = np.shape(config)[0]
    mag = np.sum(config)/(N*N)
    return abs(mag)

# Погрешность намагниченности
def standardDeviation(M, M_err):
    n = len(M_err)
    sigma_M = 0
    for i in range(n):
        sigma_M += (M - M_err[i])**2
    return np.sqrt(sigma_M/n/(n-1))

# Метод jackknife для оценки погрешности восприимчивости
def jackknife(X, M_err, M_err2, N, beta):
    n = len(M_err)
    X_jk = np.zeros(n)
    sigma_X = 0
    # Вычисление X_i без M_i
    for i in range(n):
        M1_err,M2_err = 0,0
        M1_err = np.sum(np.delete(M_err, i))
        M2_err = np.sum(np.delete(M_err2, i))
        X_jk[i] = (M2_err/(n-1) - M1_err/(n-1)*M1_err/(n-1))*beta*N**2
    #вычисление суммы (X_i - X)^2
    for i in range(n):
        sigma_X += (X_jk[i] - X)**2
    return np.sqrt(sigma_X)

# Создание решётки блочных спинов, b - параметр рескейлинга
# В данной работе используется только b=2, но функция разбивает
# на блоки произвольного размера
def rescale_config(config, b=2):
    m = np.shape(config)[0]
    rescaled_config = np.zeros((m//b, m//b))
    for i in range(m//b):
        for j in range(m//b):
            # Правило большиства (Majority rule)
            spin_sum = np.sum(config[b*i:b*i + b, b*j:b*j + b])
            if spin_sum != 0:
                rescaled_config[i,j] = np.sign(spin_sum)
            else:
                rescaled_config[i,j] = 2*np.random.randint(2) - 1
    return rescaled_config

'''
Главная вычислительная функция, реализующая метод Монте-Карло.
eqSteps - число иетраций для получения равновесного состояния.
wolffSteps - число итераций усреднения.
'''
def wolffcalc(T, N, eqSteps, wolffSteps, i_border, j_border):
    N_resc = 2*N
    N_sm = N//2
    averageSteps = wolffSteps
    errSteps = wolffSteps
    config = wolff.initialstate(N, i_border, j_border)
    config_r = wolff.initialstate(N_resc, i_border, j_border)
    config_sm = wolff.initialstate(N_sm, i_border, j_border)
    M1 = M2 = M1_r = M2_r = M1_resc = M2_resc = M1_d_resc = M2_d_resc = M1_sm = M2_sm = 0
    M1_err, M1_r_err, M1_resc_err, M1_d_resc_err, M1_sm_err = np.zeros(errSteps), np.zeros(errSteps), np.zeros(errSteps), np.zeros(errSteps), np.zeros(errSteps)
    M1_err2, M1_r_err2, M1_resc_err2, M1_d_resc_err2, M1_sm_err2 = np.zeros(errSteps), np.zeros(errSteps), np.zeros(errSteps), np.zeros(errSteps), np.zeros(errSteps)
    beta=1.0/T
    prob = 1 - np.exp(-2*beta)
    
    for _ in range(eqSteps):
        wolff.wolffmove(config, prob, i_border, j_border)
        wolff.wolffmove(config_r, prob, i_border, j_border)

    for i in range(wolffSteps):
        wolff.wolffmove(config, prob, i_border, j_border)
        wolff.wolffmove(config_r, prob, i_border, j_border)
        resc = rescale_config(config_r)
        d_resc = rescale_config(resc)
        config_sm = rescale_config(config)
        
        Mag = calcMag(config)
        Mag_r = calcMag(config_r)
        Mag_resc = calcMag(resc)
        Mag_d_resc = calcMag(d_resc)
        Mag_sm = calcMag(config_sm)

        # Вычисление наблюдаемых
        M1 += Mag
        M2 += Mag*Mag
        M1_r += Mag_r
        M2_r += Mag_r*Mag_r
        M1_resc += Mag_resc
        M2_resc += Mag_resc*Mag_resc
        M1_d_resc += Mag_d_resc
        M2_d_resc += Mag_d_resc*Mag_d_resc
        M1_sm += Mag_sm
        M2_sm += Mag_sm*Mag_sm

        # Вычисление ошибок
        M1_err[i] = Mag
        M1_err2[i] = Mag*Mag
        M1_r_err[i] = Mag_r
        M1_r_err2[i] = Mag_r*Mag_r
        M1_resc_err[i] = Mag_resc
        M1_resc_err2[i] = Mag_resc*Mag_resc
        M1_d_resc_err[i] = Mag_d_resc
        M1_d_resc_err2[i] = Mag_d_resc*Mag_d_resc
        M1_sm_err[i] = Mag_sm
        M1_sm_err2[i] = Mag_sm*Mag_sm

    # Вычисление наблюдаемых
    M = M1/averageSteps
    X = (M2/averageSteps - M1/averageSteps*M1/averageSteps)*beta*N**2
    M_r = M1_r/averageSteps
    X_r = (M2_r/averageSteps - M1_r/averageSteps*M1_r/averageSteps)*beta*N_resc**2
    M_resc = M1_resc/averageSteps
    X_resc = (M2_resc/averageSteps - M1_resc/averageSteps*M1_resc/averageSteps)*beta*N**2
    M_d_resc = M1_d_resc/averageSteps
    X_d_resc = (M2_d_resc/averageSteps - M1_d_resc/averageSteps*M1_d_resc/averageSteps)*beta*N_sm**2
    M_sm = M1_sm/averageSteps
    X_sm = (M2_d_resc/averageSteps - M1_d_resc/averageSteps*M1_d_resc/averageSteps)*beta*N_sm**2

    # Вычисление ошибок
    M_err = standardDeviation(M, M1_err)
    X_err = jackknife(X, M1_err, M1_err2, N, beta)
    M_r_err = standardDeviation(M_r, M1_r_err)
    X_r_err = jackknife(X_r, M1_r_err, M1_r_err2, N_resc, beta)
    M_resc_err = standardDeviation(M_resc, M1_resc_err)
    X_resc_err = jackknife(X_resc, M1_resc_err, M1_resc_err2, N, beta)
    M_d_resc_err = standardDeviation(M_d_resc, M1_d_resc_err)
    X_d_resc_err = jackknife(X_d_resc, M1_d_resc_err, M1_d_resc_err2, N_sm, beta)
    M_sm_err = standardDeviation(M_sm, M1_sm_err)
    X_sm_err = jackknife(X_sm, M1_sm_err, M1_sm_err2, N_sm, beta)

    return np.array([T, M, M_r, M_resc, M_d_resc, M_sm, X, X_r, X_resc, X_d_resc, X_sm,
                     M_err, M_r_err, M_resc_err, M_d_resc_err, M_sm_err, X_err, X_r_err, X_resc_err, X_d_resc_err, X_sm_err])

# Граничные условия:
# 1 - фиксированные спины s = (+1)
# 0 - непереодические (на квадрате)
# -1 - переодические (на торе)

# Объявление объектов MPI:
world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()

# Условия моделирования

nt = 10 # число значений температуры
N = 64 # число спинов на стороне решётки, т.е. решёткаразмера N*N
eqSteps = 2**14
wolffSteps = 2**16
i_border, j_border = 0, 0
T = np.linspace(2.26, 2.28, nt) # Массив температур

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

'''
Процесс ранга 0 собирает данные, посчитанные им самим
и остальными процессами, после чего записывает их в
файл critical_exponents_data.txt для последующей обработки
инструкцией critical_exponents_calc.py
'''
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
    f = open('critical_exponents\\critical_exponents_data.txt', 'w')

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