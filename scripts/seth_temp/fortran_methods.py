import numpy as np
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS:
NUM_ZETA_INTERVALS = 800 # number of zeta intervals, length of the n arrays - 1
LEVEL = 1 # which energy (e-val) and u bar set (e-vec) to use
ZETA_S = 0.1
ZETA_MAX = 150
DELTA = ZETA_MAX/(NUM_ZETA_INTERVALS + 1)
ZETA_VALS = np.arange(0, ZETA_MAX, DELTA)
N_MAX = len(ZETA_VALS)
ITERATIONS = 30 # how many times to run through the equations

G_GRAV = 6.7e-39
M_MASS = 8.2e10
A_BOHR = 1/(G_GRAV*M_MASS**3)

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
    g_00_array = np.exp(2*A_array)
    g_rr_array = np.exp(2*B_array)

    coeff_matrix = np.zeros((N_MAX, N_MAX))
    Cs, Ds, Fs = kg_find_coeffs(A_array, B_array)
    #print(f"all C's: {Cs}\nall D's: {Ds}\nall F's: {Fs}")
    for n in range(0, N_MAX):
        #C_n, D_n = kg_find_coeff(n+1, DELTA, A, B)
        coeff_matrix[n, n] = Cs[n]
        if n != N_MAX-1:
            coeff_matrix[n, n+1] = Ds[n]
        if n != 0:
            coeff_matrix[n, n-1] = Fs[n]
    
    lambdas_all, u_bars_all = np.linalg.eig(coeff_matrix)
    epsilons = lambdas_all/(1 + np.sqrt(1 + ZETA_S*lambdas_all/2))
    #print(f"    all epsilon e-vals: {epsilons}\n")
    #print(f"    index for smallest epsilon: {np.argmin(epsilons)}")
    #epsilon = epsilons[LEVEL]
    #u_bars = u_bars_all[:, LEVEL]
    epsilon = np.min(epsilons)
    u_bars = u_bars_all[:, np.argmin(epsilons)]
    
    u_tilde = np.sqrt(g_00_array/g_rr_array)*u_bars
    norm = np.sum(g_rr_array*u_tilde**2/np.sqrt(g_00_array))
    u_tilde /= np.sqrt(norm*DELTA)
    print(f"NORMALIZATION CHECK: {np.sum(u_tilde)}\n")

    u_bars = np.sqrt(g_rr_array/g_00_array)*u_tilde
    #normalize and set boundary conditions for u bar
    #norm = np.sum(np.sqrt(g_00_array)*u_bars**2) * DELTA
    #u_bars /= np.sqrt(norm)
    #u_bars[0]  = u_bars[N_MAX-1] = 0
    #u_bars /= np.sum(u_bars)
    return epsilon, u_bars

def gr_initialize_metric():
    a_array = np.zeros(N_MAX)
    b_array = np.zeros(N_MAX)
    g_00_array = np.ones(N_MAX)
    g_rr_array = np.ones(N_MAX)
    greater_idx = ZETA_VALS > ZETA_S
    a_array[greater_idx] = np.log(1-ZETA_S/ZETA_VALS[greater_idx])/2
    b_array[greater_idx] = -np.log(1-ZETA_S/ZETA_VALS[greater_idx])/2
    g_00_array[greater_idx] = 1 - ZETA_S/ZETA_VALS[greater_idx]
    g_rr_array[greater_idx] = 1/g_00_array[greater_idx]

    h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00_array/g_rr_array))

    return a_array, b_array, h_tilde


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
    #print(f"square of derivatives of R~: {derivatives**2}")
    return R_tildes, derivatives

def gr_find_Rtilde2_dRtilde2(u_bars, A_array, B_array, h_tilde):
    g_00 = np.exp(2*A_array)
    g_rr = np.exp(2*B_array)
    #h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00/g_rr))
    u_tilde = u_bars*np.sqrt(g_00/g_rr)
    
    R_tilde2 = np.zeros(N_MAX)
    dR_tilde2 = np.zeros(N_MAX)
    
    for i, zeta_n in enumerate(ZETA_VALS):
        if zeta_n != 0:
            R_tilde2[i] = (u_bars[i]**2/zeta_n**2)*np.sqrt(g_00[i]/g_rr[i])
            if i != N_MAX-1:
                dR_tilde2[i] = (u_tilde[i+1] - u_tilde[i-1])/(2*DELTA*h_tilde[i]) - u_tilde[i]*(h_tilde[i+1]-h_tilde[i-1])/(2*DELTA*h_tilde[i]**2)
                dR_tilde2[i] = dR_tilde2[i]**2
    return R_tilde2, dR_tilde2, u_tilde

# used in RK2
def gr_find_AB_slope(A_current, B_current, n, epsilon, Rt2, dRt2):
    '''
    Finds the derivatives of parameters A and B with respect to zeta,
    where A and B correspond to the metric components g00 and grr. 
    Used in the 2nd order Runge-Kutta ODE solver to get points 
    for all A(zeta) and B(zeta).
    '''
    zeta = n*DELTA
    common_term = ((ZETA_S**2)*zeta/8)*(dRt2) + (ZETA_S*zeta/4)*((1 + ZETA_S*epsilon/2)**2)*(np.exp(2*B_current-2*A_current))*(Rt2)
    if zeta != 0:
        slope_A = (np.exp(2*B_current) - 1)/(2*zeta) - ((ZETA_S*zeta)/4)*np.exp(2*B_current)*(Rt2) + common_term
        slope_B = -(np.exp(2*B_current) - 1)/(2*zeta) + ((ZETA_S*zeta)/4)*np.exp(2*B_current)*(Rt2) + common_term
    else:
        slope_A = 0
        slope_B = 0
        
    return slope_A, slope_B

def gr_BC(u_tilde, a_array, b_array):
    '''
    Return end values for A and B
    '''
    g00 = np.exp(2*a_array)
    grr = np.exp(2*b_array)

    # create mu, dmu functions
    dmu_array = np.sqrt(grr/g00)*u_tilde**2
    mu_tilde_end = 0
    for n in range(N_MAX-1):
        mu_tilde_end += DELTA*(dmu_array[n]+dmu_array[n+1])/2
    A_end = np.log(1-ZETA_S*mu_tilde_end/ZETA_VALS[N_MAX-1])/2
    B_end = -A_end
    return A_end, B_end

def gr_RK2(epsilon, Rt_array, dRt_array, u_tilde, A_array, B_array):
    '''
    Uses 2nd order Runge-Kutta ODE method to solve arrays
    for A and B. Returns two numpy arrays, for A and B values respectively.
    '''
    #A_array = np.zeros(N_MAX)
    #B_array = np.zeros(N_MAX)
    '''
    for n in range(N_MAX-1):
        A_n = A_array[n]
        B_n = B_array[n]
        Rt_n = Rt_array[n]
        dRt_n = dRt_array[n]
        slope_A_n, slope_B_n = gr_find_AB_slope(A_n, B_n, n, epsilon, Rt_n, dRt_n)
        
        A_temp = A_n + DELTA*slope_A_n
        B_temp = B_n + DELTA*slope_B_n
        slope_A_temp, slope_B_temp = gr_find_AB_slope(A_temp, B_temp, n+1, epsilon, Rt_array[n+1], dRt_array[n+1])
        
        # RK2 method
        A_array[n+1] = A_n + (DELTA/2)*(slope_A_n + slope_A_temp)
        #B_array[n+1] = B_n + (DELTA/2)*(slope_B_n + slope_B_temp)
    '''
    A_array[N_MAX-1], B_array[N_MAX-1] = gr_BC(u_tilde, A_array, B_array)
    for n in range(N_MAX-1, 0, -1):
        A_n = A_array[n]
        B_n = B_array[n]
        Rt_n = Rt_array[n]
        dRt_n = dRt_array[n]
        slope_A_n, slope_B_n = gr_find_AB_slope(A_n, B_n, n, epsilon, Rt_n, dRt_n)
        
        A_temp = A_n - DELTA*slope_A_n
        B_temp = B_n - DELTA*slope_B_n
        slope_A_temp, slope_B_temp = gr_find_AB_slope(A_temp, B_temp, n-1, epsilon, Rt_array[n-1], dRt_array[n-1])
        
        # RK2 method
        A_array[n-1] = A_n - (DELTA/2)*(slope_A_n + slope_A_temp)
        B_array[n-1] = B_n - (DELTA/2)*(slope_B_n + slope_B_temp)
    A_array[0] = 0
    B_array[0] = 0
    return A_array, B_array

# Main function that brings it together
def main():
    #initialize dynamic variables
    #A_array = np.zeros(N_MAX)
    #B_array = np.zeros(N_MAX)

    A_array, B_array, h_tilde = gr_initialize_metric()


    #temp vars for plotting
    u_bar_plot_vals = np.zeros((ITERATIONS, N_MAX))
    epsilon_plot_vals = np.zeros(ITERATIONS)

    #iterate through equations until convergence
    for i in range(ITERATIONS):
        print(f"\n\n----- In iteration number {i+1}:\n")
        epsilon, u_bar_array = kg_find_epsilon_u(A_array, B_array)
        #R_tilde_array, dR_tilde_array = gr_find_R_tildes_and_primes(u_bar_array, ZETA_VALS, A_array)
        R_tilde2, dR_tilde2, u_tilde = gr_find_Rtilde2_dRtilde2(u_bar_array, A_array, B_array, h_tilde)
    
        A_array, B_array = gr_RK2(epsilon, R_tilde2, dR_tilde2, u_tilde, A_array, B_array)
        u_bar_plot_vals[i, :] = u_bar_array

        print(f"    Calculated (lowest) epsilon value: {epsilon}\n")
        epsilon_plot_vals[i] = epsilon
        #print(f"    A vals: {A_array}\nB vals: {B_array}\n")

        # recalculate metric elements and h_tilde
        h_tilde[0] = 0
        g_00_array = np.exp(2*A_array)
        g_rr_array = np.exp(2*B_array)
        h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00_array/g_rr_array))

    g_00_array = np.exp(2*A_array)
    g_rr_array = np.exp(2*B_array)

    #u_bars_final = u_bar_plot_vals[ITERATIONS-1, :]
    #u_plot_final = u_bars_final*np.sqrt(g_00_array/(g_rr_array*A_BOHR))
    #u_plot_final = u_plot_final*np.sqrt(A_BOHR*g_rr_array/np.sqrt(g_00_array))
    plt.figure(1)
    plt.plot(ZETA_VALS,A_array, color="blue")
    plt.plot(ZETA_VALS, B_array, color="red")
    plt.xlim(0, 20)
    #plt.plot(range(ITERATIONS), epsilon_plot_vals)
    plt.grid(True)

    plt.figure(2)
    plt.plot(ZETA_VALS, abs(u_bar_array))
    plt.xlim(0, 20)
    plt.grid(True)
    plt.show()

_ = main()