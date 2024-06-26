import numpy as np
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS:
NUM_ZETA_INTERVALS = 800 # number of zeta intervals, length of the n arrays - 1
ZETA_S_VALS = [0.3]
ZETA_MAX = 80
DELTA = ZETA_MAX/(NUM_ZETA_INTERVALS + 1)
ZETA_VALS = np.arange(0, ZETA_MAX, DELTA)
N_MAX = len(ZETA_VALS)
ITERATIONS = 20 # how many times to run through the equations
MAX_ITERATIONS = 40 # how many times to run through the equations
TOLERANCE = 1.0e-6 #level of accuracy for epsilon convergence

G_GRAV = 6.7e-39
M_MASS = 8.2e10
A_BOHR = 1/(G_GRAV*M_MASS**3)

# initial metric
def find_fixed_metric(zeta_0, zeta_s):
    g_00 = np.ones(N_MAX)
    g_rr = np.ones(N_MAX)
    outside_idx = ZETA_VALS >= zeta_0
    inside_idx = ZETA_VALS < zeta_0
    #print(f"inside={inside_idx}\noutside={outside_idx}")
    def metric_f(zeta_array):
        return 1 - zeta_s*(zeta_array**2)/(zeta_0**3)
    
    g_00[inside_idx] = (1/4)*(3*np.sqrt(metric_f(zeta_0)) - np.sqrt(metric_f(ZETA_VALS[inside_idx])))**2
    g_00[outside_idx] = 1 - zeta_s/(ZETA_VALS[outside_idx])
    g_rr[outside_idx] = 1/g_00[outside_idx]
    g_rr[inside_idx] = 1/metric_f(ZETA_VALS[inside_idx])
    
    h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00/g_rr))

    A_out = np.log(g_00)/2
    B_out = np.log(g_rr)/2
    return A_out, B_out, h_tilde

# Klein Gordon equation solver ----------------------------------------------------
def kg_find_coeffs(A_array, B_array, h_tilde, zeta_s):
    '''
    Finds all the coefficients for u bar values according to
    the Klein Gordon equation, using finite difference
    approximations. Values from this function will 
    be put into matrix to find eigenvalues for energy (epsilon).

    Parameters:
    A_array: a 1d array of meteric values corre.

    Returns:
    three 1d arrays of constants
    '''
    g_00_array = np.exp(2*A_array)
    g_rr_array = np.exp(2*B_array)
    c_consts = np.zeros(N_MAX)
    d_consts = np.zeros(N_MAX)
    f_consts = np.zeros(N_MAX)

    hf_out = np.zeros(N_MAX)
    for n, zeta_n in enumerate(ZETA_VALS):
        g_frac = g_00_array[n]/g_rr_array[n]
        # fill C's:
        if zeta_n == zeta_s:
            print("\n Current zeta is the same as zeta_s (Bad!)\n")
        if n != 0 and n != N_MAX-1:
            g_frac_next = g_00_array[n+1]/g_rr_array[n+1]
            g_frac_prev = g_00_array[n-1]/g_rr_array[n-1]
            # fill D's:
            if n != N_MAX-1:
                d_consts[n] = -np.sqrt(g_frac*g_frac_next)/(DELTA**2)
            # fill F's:
            if n != 0:
                f_consts[n] = -np.sqrt(g_frac*g_frac_prev)/(DELTA**2)
            h_tilde_frac = (ZETA_VALS[n+1]*np.sqrt(np.sqrt(g_frac_next)) - 2*ZETA_VALS[n]*np.sqrt(np.sqrt(g_frac)) + ZETA_VALS[n-1]*np.sqrt(np.sqrt(g_frac_prev)))/(h_tilde[n]*DELTA**2)
            hf_out[n] = h_tilde_frac
        else:
            h_tilde_frac = 0
            hf_out[n] = h_tilde_frac
        c_consts[n] = g_frac*h_tilde_frac + (4/zeta_s)*np.exp(A_array[n])*np.sinh(A_array[n]) + 2*g_frac/(DELTA**2)
    #print(f"htildefrac: {hf_out}")
    return c_consts, d_consts, f_consts, hf_out

def kg_find_epsilon_u(A_array, B_array, h_tilde, zeta_s):
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
    Cs, Ds, Fs, hf_out = kg_find_coeffs(A_array, B_array, h_tilde, zeta_s)
    for n in range(0, N_MAX):
        #C_n, D_n = kg_find_coeff(n+1, DELTA, A, B)
        coeff_matrix[n, n] = Cs[n]
        if n != N_MAX-1:
            coeff_matrix[n, n+1] = Ds[n]
        if n != 0:
            coeff_matrix[n, n-1] = Fs[n]
    
    lambdas_all, u_bars_all = np.linalg.eig(coeff_matrix)
    lambda_min = np.min(lambdas_all)
    epsilon = lambda_min/(1 + np.sqrt(1 + zeta_s*lambda_min/2))
    u_bars = u_bars_all[:, np.argmin(lambdas_all)]
    u_tilde = np.sqrt(g_00_array/g_rr_array)*u_bars
    u_tilde[0] = 0
    u_tilde[N_MAX-1] = 0
    norm = np.sum(g_rr_array*u_tilde**2/np.sqrt(g_00_array))
    u_tilde /= np.sqrt(norm*DELTA)
    u_bars = np.sqrt(g_rr_array/g_00_array)*u_tilde

    #print(f"all C's: {Cs}\nall D's: {Ds}\nall F's: {Fs}")
    #print(f"End of u bar array: [{u_bars[N_MAX-4]}, {u_bars[N_MAX-3]}, {u_bars[N_MAX-2]}, {u_bars[N_MAX-1]}]\n")
    #print(f"NORMALIZATION CHECK: {np.trapz(u_bars, ZETA_VALS)}\n")
    #print(f"minimum lambda: {lambda_min}")

    return epsilon, u_bars

# General Relativity Metric Solver ----------------------------------------------
def gr_find_Rtilde2_dRtilde2(u_bars, A_array, B_array, h_tilde):
    g_00 = np.exp(2*A_array)
    g_rr = np.exp(2*B_array)
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
def gr_find_AB_slope(A_current, B_current, n, epsilon, Rt2, dRt2, zeta_s):
    '''
    Finds the derivatives of parameters A and B with respect to zeta,
    where A and B correspond to the metric components g00 and grr. 
    Used in the 2nd order Runge-Kutta ODE solver to get points 
    for all A(zeta) and B(zeta).
    '''
    zeta = n*DELTA
    common_term = ((zeta_s**2)*zeta/8)*(dRt2) + (zeta_s*zeta/4)*((1 + zeta_s*epsilon/2)**2)*(np.exp(2*B_current-2*A_current))*(Rt2)
    if zeta != 0:
        slope_A = (np.exp(2*B_current) - 1)/(2*zeta) - ((zeta_s*zeta)/4)*np.exp(2*B_current)*(Rt2) + common_term
        slope_B = -(np.exp(2*B_current) - 1)/(2*zeta) + ((zeta_s*zeta)/4)*np.exp(2*B_current)*(Rt2) + common_term
    else:
        slope_A = 0
        slope_B = 0
        
    return slope_A, slope_B

def gr_RK2(epsilon, Rt_array, dRt_array, A, B, zeta_s, A_start):
    '''
    Uses 2nd order Runge-Kutta ODE method to solve arrays
    for A and B. Returns two numpy arrays, for A and B values respectively.
    '''
    A_array = A
    B_array = B
    A_array[0] = A_start
    B_array[0] = 0
    for n in range(N_MAX-1):
        A_n = A_array[n]
        B_n = B_array[n]
        Rt_n = Rt_array[n]
        dRt_n = dRt_array[n]
        slope_A_n, slope_B_n = gr_find_AB_slope(A_n, B_n, n, epsilon, Rt_n, dRt_n, zeta_s)
        A_temp = A_n + DELTA*slope_A_n
        B_temp = B_n + DELTA*slope_B_n
        slope_A_temp, slope_B_temp = gr_find_AB_slope(A_temp, B_temp, n+1, epsilon, Rt_array[n+1], dRt_array[n+1], zeta_s)
        
        # RK2 method
        A_array[n+1] = A_n + (DELTA/2)*(slope_A_n + slope_A_temp)
        B_array[n+1] = B_n + (DELTA/2)*(slope_B_n + slope_B_temp)
    
    return A_array, B_array

def find_AB_root(x1, x2, fx1, fx2, tol):
    '''
    Uses secant method to find a new guess for x, to be used
    with RK2 in finding an initial condition for A.

    params:
    x1: (float) first initial A guess
    x2: (float) second initial A guess
    fx1: (float) A end based off of x1
    fx2: (float) B end based off of x2
    tol: (float) tolerance for checking for convergence

    returns:
    x1_new: (float) either x1 or x2, whichever has smaller abs(f)
    x2_new: (float) zero intercept and new guess
    meets_tol: (bool) whether or not tolerance is met
    '''
    meets_tol = False
    if abs(fx1) < abs(fx2):
        x1_new = x1
    else:
        x1_new = x2
    
    x2_new = x2 - fx2*(x2 - x1)/(fx2 - fx1)
    if abs(x2_new - x1_new) <= tol*(abs(x2_new) + abs(x1_new))/2:
        meets_tol = True
    return x1_new, x2_new, meets_tol

# Main function that brings it together
def main():
    for j in ZETA_S_VALS:
        zeta_s = j
        A_array, B_array, h_tilde = find_fixed_metric(10, zeta_s)
        A_start = A_array[0]
        epsilon = -1 # initial guess
        AB_sum = 1

        #iterate through equations until convergence
        for i in range(MAX_ITERATIONS):
            print(f"\n\n----- In iter. num. {i+1}, zeta_s={zeta_s}:\n")
            prev_epsilon = epsilon # used to check for epsilon convergence
            
            # Loop through Klein Gordon and Metric equations
            epsilon, u_bar_array = kg_find_epsilon_u(A_array, B_array, h_tilde, zeta_s)
            R_tilde2, dR_tilde2, u_tilde = gr_find_Rtilde2_dRtilde2(u_bar_array, A_array, B_array, h_tilde)
            
            meets_tol = False
            A_start1 = -0.1
            A_start2 = -0.5
            AB_adjustments = 0
            tol = 1e-6
            while meets_tol == False and AB_adjustments < 50:
                AB_adjustments+=1
                A_array1, B_array1 = gr_RK2(epsilon, R_tilde2, dR_tilde2, A_array, B_array, zeta_s, A_start1)
                fx1 = A_array1[N_MAX-1] + B_array1[N_MAX-1]
                A_array2, B_array2 = gr_RK2(epsilon, R_tilde2, dR_tilde2, A_array, B_array, zeta_s, A_start2)
                fx2 = A_array2[N_MAX-1] + B_array2[N_MAX-1]
                A_start1, A_start2, meets_tol = find_AB_root(A_start1, A_start2, fx1, fx2, tol)
                print(f"        New A_start guess: {A_start2}")
            A_array = A_array2
            B_array = B_array2
            A_start = A_start2
            print(f"Runge kutta done, took {AB_adjustments} adjustments")
            
            # recalculate metric elements and h_tilde
            h_tilde[0] = 0
            g_00_array = np.exp(2*A_array)
            g_rr_array = np.exp(2*B_array)
            h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00_array/g_rr_array))

            #A_schw = np.log(1 - zeta_s/ZETA_VALS[N_MAX-1])/2
            #B_schw = np.log(1/(1 - zeta_s/ZETA_VALS[N_MAX-1]))/2
            A_end = A_array[N_MAX-1]
            B_end = B_array[N_MAX-1]

            AB_sum = A_array[N_MAX-1]+B_array[N_MAX-1] # needs to be zero
            print(f"A_end + B_end = {AB_sum}, A_0 = {A_start}")
            #print(f"A start: {A_start}\nA end: {A_array[N_MAX-1]}, B end: {B_array[N_MAX-1]}")
            #print(f"A schw: {A_schw}, B schw: {B_schw}, schw. sum: {A_schw + B_schw}")
            print(f"Calculated (lowest) epsilon value: {epsilon}\n")

            #A_start -= AB_sum

            #A_diff = A_end - A_schw
            #print(f"A_diff: {A_diff}")
            #A_start -= A_diff

            # check for epsilon convergence within tolerance
            if abs(epsilon - prev_epsilon) <= TOLERANCE:
                iter_to_tolerance = i+1
                print(f"Tolerance met! Took {iter_to_tolerance} iteration(s) for this adjustment")
                break
        print(f"\n ----------- Iterations and adjustments finished ------------")
        print(f"In {AB_adjustments} adjustments (zeta_s={zeta_s}) the calculated epsilon is {epsilon}, accurate to {TOLERANCE}")
        g_00_array = np.exp(2*A_array)
        g_rr_array = np.exp(2*B_array)

        plt.figure(1)
        plt.plot(ZETA_VALS, u_bar_array, color='blue', alpha=0.5, marker='.', label="$u_{bar}$" + f", A_end+B_end={AB_sum}")
        plt.xlim(0, 20)
        plt.xlabel("$\zeta$")
        plt.legend()
        plt.grid(True)
        plt.title(f"$\zeta_s$={zeta_s}, $\epsilon$={epsilon}\nTol: {TOLERANCE}\nZ_max={ZETA_MAX}, Z_int={NUM_ZETA_INTERVALS}, $A_0$={A_start}")

        plt.figure(2)
        plt.plot(ZETA_VALS, A_array, color='blue', alpha=0.5, marker='.', label="A")
        plt.plot(ZETA_VALS, B_array, color='red', alpha=0.5, marker='.', label="B")
        plt.xlim(0, 20)
        plt.xlabel(f"$\zeta$")
        plt.legend()
        plt.grid(True)
        plt.title(f"$\zeta_s$={zeta_s}, $\epsilon$={epsilon}\nTol: {TOLERANCE}\nZ_max={ZETA_MAX}, Z_int={NUM_ZETA_INTERVALS}, $A_0$={A_start}")
    plt.show()

_ = main()