# ZETA_S = 0.01, del = 0.01, n = 20, A = B = 0

import numpy as np
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS:
N_MAX = 1000
LEVEL = 1 # which energy (e-val) and u bar set (e-vec) to use
ZETA_S = 0.5
ZETA_MAX = 10

DELTA = ZETA_MAX/(N_MAX + 1)

# Klein Gordon equation solver ----------------------------------------------------
def kg_find_coeff(n, delta, A, B):
    '''
    Finds the nth coefficients for u bar values according to
    the Klein Gordon equation, using finite difference
    approximations. Values from this function will 
    be put into matrix to find eigenvalues for energy (epsilon).
    '''

    g_00 = np.exp(2*A)
    g_rr = np.exp(2*B)
    zeta_n = n*delta
    if zeta_n != ZETA_S:
        h_tilde_frac = -(ZETA_S**2)/(4*(zeta_n**2)*(zeta_n**2 - ZETA_S**2))
        c_const = (g_00/g_rr)*h_tilde_frac + (4/ZETA_S)*np.exp(A)*np.sinh(A) + 2*(g_00/g_rr)/(delta**2)
        d_const = -(g_00/g_rr)/(delta**2)
    else:
        c_const = 0
        d_const = 0

    return c_const, d_const

#try n = 1
c_1, d_1 = kg_find_coeff(1, 0.01, 0, 0)
print(f"C coefficient for n=1: {c_1}")

def kg_find_epsilon_u(n_size, A, B):
    '''
    Creates a matrix of u bar coefficients in the Klein Gordon
    equation, based off of n steps of zeta (rescaled radius).
    returns a single rescaled energy (epsilon) and 1D
    array of u bar values, both according to global
    LEVEL parameter.
    '''
    
    coeff_matrix = np.zeros((n_size, n_size))
    for n in range(0, n_size):
        C_n, D_n = kg_find_coeff(n+1, DELTA, A, B)
        coeff_matrix[n, n] = C_n
        if n != n_size-1:
            coeff_matrix[n, n+1] = D_n
        if n != 0:
            coeff_matrix[n, n-1] = D_n
    
    lambdas_all, u_bars_all = np.linalg.eig(coeff_matrix)
    lambda_spec = lambdas_all[LEVEL]
    u_bars = u_bars_all[:, LEVEL]
    epsilon = lambda_spec/(1 + np.sqrt(1 + ZETA_S*lambda_spec/2))
    return epsilon, u_bars

'''
d_zeta = 0.0001
zetas = np.arange(0, 0.009, d_zeta)
y_vals, u_bars = kg_find_epsilons(len(zetas), d_zeta, 0, 0)
print(f"number of zetas: {len(zetas)}")
print(f"epsilon values: {y_vals}")
print(f"u bar values: {u_bars}")
plt.plot(zetas, y_vals)
plt.grid(True)
plt.show()
'''
# General Relativity Metric Solver ----------------------------------------------

# uses numpy arrays, u_bars and zetas must be same size
# not sure if this is right
def gr_find_R_tildes_and_primes(u_bars, zetas, g_00):
    R_tildes = (u_bars)*np.sqrt(g_00)/zetas
    size = len(R_tildes)
    derivatives = np.zeros(size)
    for i in range(size):
        if i != 0 or i != size-1:
            derivatives[i] = (R_tildes[i+1] - R_tildes[i-1])/(zetas[i+1] - zetas[i-1])
        else:
            derivatives[i] = 0
    return R_tildes, derivatives



def gr_find_AB_slope(A_current, B_current, n, epsilon, Rt, dRt):
    '''
    Finds the derivatives of parameters A and B with respect to zeta,
    where A and B correspond to the metric components g00 and grr. 
    Used in the 2nd order Runge-Kutta ODE solver to get points 
    for all A(zeta) and B(zeta).
    '''
    zeta = n*DELTA
    common_term = ((ZETA_S**2)*zeta/8)*(dRt**2) 
    + (ZETA_S*zeta/4)*((1 + ZETA_S*epsilon/2)**2)*(np.exp(2*B_current-2*A_current))*(Rt**2)

    slope_A = (np.exp(2*B_current) - 1)/(2*zeta) - ((ZETA_S*zeta)/4)*np.exp(2*B_current)*(Rt**2) + common_term
    slope_B = -(np.exp(2*B_current) - 1)/(2*zeta) + ((ZETA_S*zeta)/4)*np.exp(2*B_current)*(Rt**2) + common_term
    return slope_A, slope_B

print(f"some slope: {gr_find_AB_slope(0, 0, 1, 1, 1, 1)}")