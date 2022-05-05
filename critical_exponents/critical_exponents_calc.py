from scipy import optimize
import numpy as np
from numpy.random import rand
from matplotlib import pyplot as plt
from matplotlib import ticker
plt.style.use('seaborn-poster')

def calc_nu(N, T, M, M_r, M_resc, M_doub_resc, M_sm, X, X_r, X_resc, X_doub_resc, X_sm,
             M_err, M_r_err, M_resc_err, M_doub_resc_err, M_sm_err, X_err, X_r_err, X_resc_err, X_doub_resc_err, X_sm_err):
    
    N = int(N)
    # Намагниченность (ню и бета)
    n = len(T)
    weights = np.array([n/(n + 2*abs(n//2 - i)) for i in range(n)])

    M_a, M_b, M_c = np.polyfit(T, M, 2, w=weights)
    M_a_r, M_b_r, M_c_r = np.polyfit(T, M_r, 2, w=weights)
    M_a_resc, M_b_resc, M_c_resc = np.polyfit(T, M_resc, 2, w=weights)
    M_a_doub_resc, M_b_doub_resc, M_c_doub_resc = np.polyfit(T, M_doub_resc, 2, w=weights)
    M_a_sm, M_b_sm, M_c_sm = np.polyfit(T, M_sm, 2, w=weights)

    M_curve = lambda x: M_a*x**2 + M_b*x + M_c
    M_curve_r = lambda x: M_a_r*x**2 + M_b_r*x + M_c_r
    M_curve_resc = lambda x: M_a_resc*x**2 + M_b_resc*x + M_c_resc
    M_curve_doub_resc = lambda x: M_a_doub_resc*x**2 + M_b_doub_resc*x + M_c_doub_resc


    M_curve_inter = lambda y: M_curve(y) - M_curve_resc(y)

    M_T_c = optimize.newton(M_curve_inter, 2.268)

    nu = np.log(2)/np.log((2*M_a_resc*M_T_c + M_b_resc)/(2*M_a*M_T_c + M_b))
    nu_doub = np.log(2)/np.log((2*M_a_doub_resc*M_T_c + M_b_doub_resc)/(2*M_a_sm*M_T_c + M_b_sm))
    beta = np.log((M_a_resc*M_T_c**2 + M_b_resc*M_T_c + M_c_resc)/(M_a_r*M_T_c**2 + M_b_r*M_T_c + M_c_r))/np.log(2)*nu
    beta_doub = np.log((M_a_doub_resc*M_T_c**2 + M_b_doub_resc*M_T_c + M_c_doub_resc)/(M_a_resc*M_T_c**2 + M_b_resc*M_T_c + M_c_resc))/np.log(2)*nu

    # Восприимчивость (гамма)

    X_a, X_b, X_c, X_d = np.polyfit(T, X, 3, w=weights)
    X_a_r, X_b_r, X_c_r, X_d_r = np.polyfit(T, X_r, 3, w=weights)
    X_a_resc, X_b_resc, X_c_resc, X_d_resc = np.polyfit(T, X_resc, 3, w=weights)
    X_a_doub_resc, X_b_doub_resc, X_c_doub_resc, X_d_doub_resc = np.polyfit(T, X_doub_resc, 3, w=weights)
    X_a_sm, X_b_sm, X_c_sm, X_d_sm = np.polyfit(T, X_sm, 3, w=weights)

    X_curve = lambda x: X_a*x**3 + X_b*x**2 + X_c*x + X_d
    X_curve_r = lambda x: X_a_r*x**3 + X_b_r*x**2 + X_c_r*x + X_d_r
    X_curve_resc = lambda x: X_a_resc*x**3 + X_b_resc*x**2 + X_c_resc*x + X_d_resc
    X_curve_intersect = lambda y: X_curve(y) - X_curve_resc(y)

    X_T_c = optimize.newton(X_curve_intersect, 2.268)
    
    gamma = -np.log((X_a_resc*X_T_c**3 + X_b_resc*X_T_c**2 + X_c_resc*X_T_c + X_d_resc)/(X_a_r*X_T_c**3 + X_b_r*X_T_c**2 + X_c_r*X_T_c + X_d_r))/np.log(2)*nu
    gamma_doub = -np.log((X_a_doub_resc*X_T_c**3 + X_b_doub_resc*X_T_c**2 + X_c_doub_resc*X_T_c + X_d_doub_resc)/(X_a_resc*X_T_c**3 + X_b_resc*X_T_c**2 + X_c_resc*X_T_c + X_d_resc))/np.log(2)*nu

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))

    # Построение графиков
    fig_M, ax = plt.subplots(figsize=(18, 12))

    ax.scatter(T, M, s=50, marker='o', color='IndianRed')
    ax.plot(T, M_curve(T), color='IndianRed', label=r'$M(N = %s)$' %N)
    ax.errorbar(T, M, yerr = M_err, color='IndianRed', linestyle = 'None', capsize = 5, markeredgewidth=2)

    ax.scatter(T, M_r, s=50, marker='o', color='IndianRed')
    ax.plot(T, M_curve_r(T), '--', color='IndianRed', label=r'$M(N = %s)$' %(2*N))
    ax.errorbar(T, M_r, yerr = M_r_err, color='IndianRed', linestyle = 'None', capsize = 5, markeredgewidth=2)

    ax.scatter(T, M_resc, s=50, marker='o', color='RoyalBlue')
    ax.plot(T, M_curve_resc(T), color='RoyalBlue', label=r'$M_{resc}(N = %s)$' %N)
    ax.errorbar(T, M_resc, yerr = M_resc_err, color='RoyalBlue', linestyle = 'None', capsize = 5, markeredgewidth=2)

    ax.set_xlabel("Температура", fontsize=24)
    ax.set_ylabel("Намагниченность", fontsize=24)
    ax.grid(linestyle='--', linewidth=1)
    ax.axis('tight')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('% 1.2f'))
    ax.legend(fontsize=24)

    fig_M.savefig("plots\\nu_exponent.png")

    fig_X, ax = plt.subplots(figsize=(18, 12))

    ax.scatter(T, X, s=50, marker='o', color='IndianRed')
    ax.plot(T, X_curve(T), color='IndianRed', label=r'$\chi(N = %s)$' %N)
    ax.errorbar(T, X, yerr = X_err, color='IndianRed', linestyle = 'None', capsize = 5, markeredgewidth=2)

    ax.scatter(T, X_r, s=50, marker='o', color='IndianRed')
    ax.plot(T, X_curve_r(T), linestyle='--', color='IndianRed', label=r'$\chi(N = %s)$' %(2*N))
    ax.errorbar(T, X_r, yerr = X_r_err, color='IndianRed', linestyle = 'None', capsize = 5, markeredgewidth=2)

    ax.scatter(T, X_resc, s=50, marker='o', color='RoyalBlue')
    ax.plot(T, X_curve_resc(T), color='RoyalBlue', label=r'$\chi_{resc}(N = %s)$' %N)
    ax.errorbar(T, X_resc, yerr = X_resc_err, color='RoyalBlue', linestyle = 'None', capsize = 5, markeredgewidth=2)

    ax.set_xlabel("Температура", fontsize=20)
    ax.set_ylabel("Восприимчивость", fontsize=20)
    ax.grid(linestyle='--', linewidth=1)
    ax.axis('tight')

    ax.legend(fontsize=18)

    fig_X.savefig("plots\\gamma_exponent.png")

    print('|------------Results-------------|\n')
    print(f'\tM_T_c = {M_T_c:.4f}')
    print(f'\tX_T_c = {X_T_c:.4f}')
    print(f'\tnu = {nu:.3f}')
    print(f'\tnu_doub = {nu_doub:.3f}')
    print(f'\tbeta = {beta:.3f}')
    print(f'\tbeta_doub = {beta_doub:.3f}')
    print(f'\tgamma = {gamma:.3f}')
    print(f'\tgamma_doub = {gamma_doub:.3f}')
    print('|-------------------------------|\n')


f = open('critical_exponents\\critical_exponents_data.txt', 'r')
N = f.readline()
lines = f.readlines()

data = np.array([line.split('; ') for line in lines])
data = np.array([np.array(line, dtype=float) for line in data])

f.close()

calc_nu(N, *data)

