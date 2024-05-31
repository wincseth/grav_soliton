import numpy as np
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS:
NUM_ZETA_INTERVALS = 100 # number of zeta intervals, length of the n arrays - 1
LEVEL = 1 # which energy (e-val) and u bar set (e-vec) to use
ZETA_S = 0.5
ZETA_MAX = 10
DELTA = ZETA_MAX/(NUM_ZETA_INTERVALS + 1)
ZETA_VALS = np.arange(0, ZETA_MAX, DELTA)
N_MAX = len(ZETA_VALS)
ITERATIONS = 20 # how many times to run through the equations
print(f"length of zeta array: {len(ZETA_VALS)}")
# Klein Gordon equation solver ----------------------------------------------------
def kg_find_coeffs(A_array, B_array):
    '''
    Finds all the coefficients for u bar values according to
    the Klein Gordon equation, using finite difference
    approximations. Values from this function will 
    be put into matrix to find eigenvalues for energy (epsilon).

    Parameters:
    A_array: a 1d array of meteric values corre..

    Returns:
    three 1d arrays of constants
    '''
    g_00_array = np.exp(2*A_array)
    g_rr_array = np.exp(2*B_array)
    c_consts = np.zeros(N_MAX)
    d_consts = np.zeros(N_MAX)
    f_consts = np.zeros(N_MAX)
    for n, zeta_n in enumerate(ZETA_VALS):
        g_frac = g_00_array[n]/g_rr_array[n]
        # fill C's:
        if zeta_n == ZETA_S:
            print("\n Current zeta is the same as zeta_s (Bad!)\n")
        if zeta_n != 0:
            h_tilde_frac = -(ZETA_S**2)/(4*(zeta_n**2)*(zeta_n**2 - ZETA_S**2))
        else:
            h_tilde_frac = 0    
        c_consts[n] = g_frac*h_tilde_frac + (4/ZETA_S)*np.exp(A_array[n])*np.sinh(A_array[n]) + 2*g_frac/(DELTA**2)
        # fill D's:
        if n != N_MAX-1:
            g_frac_next = g_00_array[n+1]/g_rr_array[n+1]
            d_consts[n] = -np.sqrt(g_frac*g_frac_next)/(DELTA**2)
        # fill F's:
        if n != 0:
            g_frac_prev = g_00_array[n-1]/g_rr_array[n-1]
            f_consts[n] = -np.sqrt(g_frac*g_frac_prev)/(DELTA**2)

    return c_consts, d_consts, f_consts

def kg_find_epsilon_u(A_array, B_array):
    '''
    Creates a matrix of u bar coefficients in the Klein Gordon
    equation, based off of n steps of zeta (rescaled radius).
    returns a single rescaled energy (epsilon) and 1D
    array of u bar values, both according to global
    LEVEL parameter.
    '''
    
    coeff_matrix = np.zeros((N_MAX, N_MAX))
    Cs, Ds, Fs = kg_find_coeffs(A_array, B_array)
    for n in range(0, N_MAX):
        #C_n, D_n = kg_find_coeff(n+1, DELTA, A, B)
        coeff_matrix[n, n] = Cs[n]
        if n != N_MAX-1:
            coeff_matrix[n, n+1] = Ds[n]
        if n != 0:
            coeff_matrix[n, n-1] = Fs[n]
    
    lambdas_all, u_bars_all = np.linalg.eig(coeff_matrix)
    u_bars = u_bars_all[:, LEVEL]
    epsilons = lambdas_all/(1 + np.sqrt(1 + ZETA_S*lambdas_all/2))
    print(f"    all epsilon e-vals: {epsilons}\n")
    #epsilon = epsilons[LEVEL]
    epsilon = np.min(epsilons)
    return epsilon, u_bars


# General Relativity Metric Solver ----------------------------------------------
# uses numpy arrays, u_bars and zetas must be same size
# not sure if this is right
def gr_find_R_tildes_and_primes(u_bars, zetas, A_array):
    '''
    Finds R~ and dR~/dzeta to be used for dA and dB derivatives.
    
    Parameters: 
        u_bars: 1d np array
        zetas: 1d np array
        A_array: 1d np array
    Returns:
        R_tildes: 1d np array
        derivatives: 1d np array
    '''
    g_00_array = np.exp(2*A_array)

    # calculate R~ and avoid division by zeros from zeta vals
    R_tildes = np.zeros_like(zetas)
    non_zeros_mask = zetas != 0
    R_tildes[non_zeros_mask] = (u_bars[non_zeros_mask]*np.sqrt(g_00_array[non_zeros_mask])/zetas[non_zeros_mask])
    
    # calculate dR~
    size = len(R_tildes)
    derivatives = np.zeros(size)
    for i in range(size):
        if i != 0 and i != size-1:
            derivatives[i] = (R_tildes[i+1] - R_tildes[i-1])/(2*DELTA)
        else:
            derivatives[i] = 0
    return R_tildes, derivatives

# used in RK2
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
    if zeta != 0:
        slope_A = (np.exp(2*B_current) - 1)/(2*zeta) - ((ZETA_S*zeta)/4)*np.exp(2*B_current)*(Rt**2) + common_term
        slope_B = -(np.exp(2*B_current) - 1)/(2*zeta) + ((ZETA_S*zeta)/4)*np.exp(2*B_current)*(Rt**2) + common_term
    else:
        slope_A = 0
        slope_B = 0
        
    return slope_A, slope_B

def gr_RK2(epsilon, Rt_array, dRt_array):
    '''
    Uses 2nd order Runge-Kutta ODE method to solve arrays
    for A and B. Returns two numpy arrays, for A and B values respectively.
    '''
    A_array = np.zeros(N_MAX)
    B_array = np.zeros(N_MAX)
    for n in range(N_MAX-1):
        A_n = A_array[n]
        B_n = B_array[n]
        Rt_n = Rt_array[n]
        dRt_n = dRt_array[n]
        slope_A_n, slope_B_n = gr_find_AB_slope(A_n, B_n, n, epsilon, Rt_n, dRt_n)
        A_temp = A_n + DELTA*slope_A_n
        B_temp = B_n + DELTA*slope_B_n
        slope_A_temp, slope_B_temp = gr_find_AB_slope(A_temp, B_temp, n+1, epsilon, Rt_n, dRt_n)
        
        # RK2 method
        A_array[n+1] = A_n + (DELTA/2)*(slope_A_n + slope_A_temp)
        B_array[n+1] = B_n + (DELTA/2)*(slope_B_n + slope_B_temp)
    
    return A_array, B_array

'''
print(f"some slope: {gr_find_AB_slope(0, 0, 1, 1, 1, 1)}")
#try n = 1
c_1, d_1 = kg_find_coeff(1, 0.01, 0, 0)
print(f"C coefficient for n=1: {c_1}")
'''

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

# Main function that brings it together
def main():
    #initialize dynamic variables
    A_array = np.zeros(N_MAX)
    B_array = np.zeros(N_MAX)

    u_bar_plot_vals = np.zeros((ITERATIONS, N_MAX))
    for i in range(ITERATIONS):
        print(f"\n\n----- In iteration number {i+1}:\n")
        epsilon, u_bar_array = kg_find_epsilon_u(A_array, B_array)
        R_tilde_array, dR_tilde_array = gr_find_R_tildes_and_primes(u_bar_array, ZETA_VALS, A_array)

        A_array, B_array = gr_RK2(epsilon, R_tilde_array, dR_tilde_array)
        u_bar_plot_vals[i, :] = u_bar_array

        print(f"    Calculated (lowest) epsilon value: {epsilon}\n")
        print(f"    A vals: {A_array}\nB vals: {B_array}\n")

    plt.plot(ZETA_VALS, u_bar_plot_vals[ITERATIONS-1, :])
    plt.show()

_ = main()