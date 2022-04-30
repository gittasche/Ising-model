import numpy as np
from collections import deque
from numpy.random import rand

'''
Создание начального состояния решётки с учётом
граничных условий. Граничные условия:
1 - фиксированные спины s = (+1)
0 - непереодические (на квадрате)
-1 - переодические (на торе)
'''
def initialstate(N, i_border, j_border):
    state = 2*np.random.randint(2, size=(N,N))-1
    if i_border == 1:
        state[0,:] = np.ones(N)
        state[N-1,:] = np.ones(N)
    if j_border == 1:
        state[:,0] = np.ones(N)
        state[:,N-1] = np.ones(N)
    return state

'''
Осноной алгоритм, реалицующий статистику, 
которой подчиняются состояния решётки.
Используется двусторонняя очередь deque,
для быстрой работы. В данном случае реализована
FIFO очередь.
'''
def wolffmove(config, prob, i_border, j_border):
    N = np.shape(config)[0]
    j_b = j_border
    i_b = i_border
    if i_b > 0: # при i_b > 0 грница строго закреплена
        i = np.random.randint(i_b, N - i_b)
    else:
        i = np.random.randint(N)
    if j_b > 0:
        j = np.random.randint(j_b, N - j_b)
    else:
        j = np.random.randint(N)
    cluster = deque()
    cluster.append(i)
    cluster.append(j)
    oldspin = config[i,j]
    newspin = -1*config[i,j]
    config[i,j] = newspin

    while len(cluster) > 0:
        # Заменив popleft на pop и поменяв местами в методах append
        # индексы i и j местами получим реализацию через LIFO стек,
        # что даёт такой же результат, как и при FIFO.
        current_i = cluster.popleft()
        current_j = cluster.popleft()
        if current_j != j_b and current_j != N-1-j_b:
            i_n = current_i
            j_n = (current_j + 1)%N
            if config[i_n,j_n] == oldspin:
                if rand() < prob:
                    cluster.append(i_n)
                    cluster.append(j_n)
                    config[i_n,j_n] = newspin
                    
        if current_j != j_b and current_j != N-1-j_b:
            i_n = current_i
            j_n = (current_j - 1)%N
            if config[i_n,j_n] == oldspin:
                if rand() < prob:
                    cluster.append(i_n)
                    cluster.append(j_n)
                    config[i_n,j_n] = newspin

        if current_i != i_b and current_i != N-1-i_b:
            i_n = (current_i + 1)%N
            j_n = current_j
            if config[i_n,j_n] == oldspin:
                if rand() < prob:
                    cluster.append(i_n)
                    cluster.append(j_n)
                    config[i_n,j_n] = newspin

        if current_i != i_b and current_i != N-1-i_b:
            i_n = (current_i - 1)%N
            j_n = current_j
            if config[i_n,j_n] == oldspin:
                if rand() < prob:
                    cluster.append(i_n)
                    cluster.append(j_n)
                    config[i_n,j_n] = newspin

    return config