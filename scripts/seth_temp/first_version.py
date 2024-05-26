# zeta_s = 0.01, del = 0.01, n = 20, A = B = 0

import numpy as np
import matplotlib.pyplot as plt

def kg_find_coeff(n, delta, A, B):
    '''
    Finds the nth coefficients for u bar values according to
    the Klein Gordon equation, using finite difference
    approximations. Values from this function will 
    be put into matrix to find eigenvalues for energy (epsilon).
    '''
    zeta_s = 0.01 # make this global in the future
    g_00 = np.exp(2*A)
    g_rr = np.exp(2*B)
    zeta_n = n*delta
    if zeta_n != zeta_s:
        h_tilde_frac = -(zeta_s**2)/(4*(zeta_n**2)*(zeta_n**2 - zeta_s**2))
        c_const = (g_00/g_rr)*h_tilde_frac + (4/zeta_s)*np.exp(A)*np.sinh(A) + 2*(g_00/g_rr)/(delta**2)
        d_const = -(g_00/g_rr)/(delta**2)
    else:
        c_const = 0
        d_const = 0

    return c_const, d_const

#try n = 1
c_1, d_1 = kg_find_coeff(1, 0.01, 0, 0)
print(f"C coefficient for n=1: {c_1}")

def kg_find_epsilons(n_size, delta, A, B):
    '''
    Creates a matrix of u bar coefficients in the Klein Gordon
    equation, based off of n steps of zeta (rescaled radius).
    returns a 1D array of rescaled energies (aka epsilons) that
    are found with the eigenvalues (aka lambdas) from the
    coefficient matrix.
    '''
    zeta_s = 0.01 #make this global in the future
    d = 1/(n_size + 1)
    coeff_matrix = np.zeros((n_size, n_size))
    for n in range(0, n_size):
        C_n, D_n = kg_find_coeff(n+1, d, A, B)
        coeff_matrix[n, n] = C_n
        if n != n_size-1:
            coeff_matrix[n, n+1] = D_n
        if n != 0:
            coeff_matrix[n, n-1] = D_n
    
    lambdas, eigenvectors = np.linalg.eig(coeff_matrix)
    epsilons = lambdas/(1 + np.sqrt(1 + zeta_s*lambdas/2))
    return epsilons

d_zeta = 0.0001
zetas = np.arange(0, 0.009, d_zeta)
y_vals = kg_find_epsilons(len(zetas), d_zeta, 0, 0)
print(f"epsilon values: {y_vals}")
plt.plot(zetas, y_vals)
plt.grid(True)
plt.show()

# uses numpy arrays, u_bars and zetas must be same size
def find_R_tildes_and_primes(u_bars, zetas, g_00):
    R_tildes = (u_bars)*np.sqrt(g_00)/zetas
    size = len(R_tildes)
    derivatives = np.zeros(size)
    for i in range(size):
        if i != 0 or i != size-1:
            derivatives[i] = (R_tildes[i+1] - R_tildes[i-1])/(zetas[i+1] - zetas[i-1])
        else:
            derivatives[i] = 0
    return R_tildes, derivatives

