import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
plt.style.use('seaborn-poster')

'''
Обработка данных программы Wolff_2Darr.py
'''

def wolffPlots(T, E, C, M, X, E_err, M_err, C_err, X_err):
    f, axs = plt.subplots(2, 2, figsize=(18, 10))
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))

    axs[0, 0].scatter(T, E, s=50, marker='o', color='IndianRed')
    axs[0, 0].errorbar(T, E, yerr = E_err, color='IndianRed', linestyle = 'None', capsize = 5, markeredgewidth=2)
    axs[0, 0].set_xlabel("Температура", fontsize=20)
    axs[0, 0].set_ylabel("Энергия", fontsize=20)   
    axs[0, 0].grid(linestyle='--', linewidth=1)     
    axs[0, 0].axis('tight')


    axs[1, 0].scatter(T, C, s=50, marker='o', color='IndianRed')
    axs[1, 0].errorbar(T, C, yerr = C_err, color='IndianRed', linestyle = 'None', capsize = 5, markeredgewidth=2)
    axs[1, 0].set_xlabel("Температура", fontsize=20)
    axs[1, 0].set_ylabel("Теплоёмкость", fontsize=20)
    axs[1, 0].grid(linestyle='--', linewidth=1)
    axs[1, 0].axis('tight')


    axs[0, 1].scatter(T, M, s=50, marker='o', color='RoyalBlue')
    axs[0, 1].errorbar(T, M, yerr = M_err, color='RoyalBlue', linestyle = 'None', capsize = 5, markeredgewidth=2)
    axs[0, 1].set_xlabel("Температура", fontsize=20)
    axs[0, 1].set_ylabel("Намагниченность", fontsize=20)
    axs[0, 1].grid(linestyle='--', linewidth=1)
    axs[0, 1].axis('tight')


    axs[1, 1].scatter(T, X, s=50, marker='o', color='RoyalBlue')
    axs[1, 1].errorbar(T, X, yerr = X_err, color='RoyalBlue', linestyle = 'None', capsize = 5, markeredgewidth=2)
    axs[1, 1].set_xlabel("Температура", fontsize=20)
    axs[1, 1].set_ylabel("Восприимчивость", fontsize=20)
    axs[1, 1].grid(linestyle='--', linewidth=1)
    axs[1, 1].axis('tight')

    plt.savefig("plots\\observables.png")

f = open('critical_exponents\\wolff_data.txt', 'r')

readed = f.readline()

lines = f.readlines()
data = np.array([line.split(',') for line in lines])
data = np.array([np.array(line, dtype=float) for line in data])

f.close()

wolffPlots(*data)