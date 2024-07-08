import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS:
NUM_ZETA_INTERVALS = 1000 # number of zeta intervals, length of zeta array - 1
ZETA_S_VALS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] # which values of zeta_s to calculate
ZETA_MAX = 50 # how far to calculate zeta to
DELTA = ZETA_MAX/(NUM_ZETA_INTERVALS + 1) # zeta array step size
ZETA_VALS = np.arange(0, ZETA_MAX, DELTA) # zeta array
N_MAX = len(ZETA_VALS)
MAX_ITERATIONS = 40 # hard cap on how many times to run through the equations
TOLERANCE = 1.0e-6 #level of accuracy for epsilon convergence

# physical parameters
G_GRAV = 6.7e-39
M_MASS = 8.2e10
A_BOHR = 1/(G_GRAV*M_MASS**3)
SAVE_PARAMS = {
    'u_bar': True,
    'AB': True,
    'R_tilde': True,
    'A_start': True,
    'epsilon': True,
    'E/m': True,
    'write_file': True,
    'custom_name': True
}

# initial metric
def find_fixed_metric(zeta_0, zeta_s):
    '''
    Makes an initial guess for the metric with schwarchild approximation
    past a specified zeta value and a formula (insert reference here)
    for before the specified zeta value.

    :params:
    zeta_0: (float) rescaled radius to switch metric definitions
    zeta_s: (float) relativistic parameter

    :returns:
    A_out: (np array) exponential factors used in time portion of metric
    B_out: (np array) exponential factors used in radius portion of metric
    h_tilde: (np array) radial function rescaling factor based off of calculated metric
    '''
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

    :params:
    A_array: (np array) exponential factors for time metric
    B_array: (np array) exponential factors for radius metric
    h_tilde: (np array) radial function rescaling factor based off of metric
    zeta_s: (float) relativistic parameter

    :returns:
    c_consts: (np array) diagonal matrix elements
    d_consts: (np array) off diagonal (right) matrix elements
    f_consts: (np array) off diagonal (left) matrix elements
    '''
    g_00_array = np.exp(2*A_array)
    g_rr_array = np.exp(2*B_array)
    c_consts = np.zeros(N_MAX)
    d_consts = np.zeros(N_MAX)
    f_consts = np.zeros(N_MAX)

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
        else:
            h_tilde_frac = 0

        c_consts[n] = g_frac*h_tilde_frac + (4/zeta_s)*np.exp(A_array[n])*np.sinh(A_array[n]) + 2*g_frac/(DELTA**2)
    return c_consts, d_consts, f_consts

def kg_find_epsilon_u(A_array, B_array, h_tilde, zeta_s):
    '''
    Creates a matrix of u bar coefficients in the Klein Gordon
    equation, based off of n steps of zeta (rescaled radius).
    returns a single rescaled energy (epsilon) and 1D
    array of u bar values, based off of the results of the
    matrix eigenvalue problem.

    :params:
    A_array: (np array) exponential factors for time metric
    B_array: (np array) exponential factors for radius metric
    h_tilde: (np array) radial function rescaling factor based off of metric
    zeta_s: (float) relativistic parameter

    :returns:
    epsilon: (float) rescaled energy
    u_bars: (np array) rescaled wave function
    '''
    g_00_array = np.exp(2*A_array)
    g_rr_array = np.exp(2*B_array)

    coeff_matrix = np.zeros((N_MAX, N_MAX))
    Cs, Ds, Fs = kg_find_coeffs(A_array, B_array, h_tilde, zeta_s)
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

    return epsilon, u_bars

# General Relativity Metric Solver ----------------------------------------------
def gr_find_Rtilde2_dRtilde2(u_bars, A_array, B_array, h_tilde):
    '''
    Calculates the square of rescaled radius function (R tilde) as well as
    the first derivative with respect to zeta. These are both used in the general
    relativistic differential equations which are solved with the Runge Kutta method.

    :params:
    u_bars: (np array) rescaled wave function
    A_array: (np array) exponential factors for time metric
    B_array: (np array) exponential factors for radius metric
    h_tilde: (np array) radial function rescaling factor based off of metric

    :returns:
    R_tilde2: (np array) R tilde squared 
    dR_tilde2: (np array) derivative of R tilde (w.r.t. zeta) squared
    '''
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
    R_tilde2[0] = R_tilde2[1]
    dR_tilde2[0] = dR_tilde2[1]
    return R_tilde2, dR_tilde2, u_tilde

# used in RK2
def gr_find_AB_slope(A_current, B_current, n, epsilon, Rt2, dRt2, zeta_s):
    '''
    Finds the derivatives of parameters A and B with respect to zeta,
    where A and B correspond to the metric components g00 and grr. 
    Used in the 2nd order Runge-Kutta ODE solver to get points 
    for all A(zeta) and B(zeta).

    :params:
    A_current: (float) specific A value
    B_current: (float) specific B value
    n: (int) index shared between A, B, and R
    epsilon: (float) rescaled energy
    Rt2: (float) specific R tilde squared
    dRt2: (float) specific dR tilde/dzeta squared
    zeta_s: (float) relativistic parameter

    :returns: 
    slope_A: (float) dA/dZeta
    slope_B: (float) dB/dZeta
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
    Uses 2nd order Runge-Kutta ODE method to solve for arrays
    for A and B (exponential metric factors), using previous A B guesses
    and initial conditions at zeta=0. 
    Returns two numpy arrays, for A and B values respectively.
    
    :params:
    epsilon: (float) rescaled energy
    Rt_array: (np array) R tilde values squared
    dRt_array: (np array) derivative of R tilde values squared
    A: (np array) inputted A values
    B: (np array) inputted B values
    zeta_s: (float) relativistic parameter
    A_start: (float) initial boundary condition for A (at zeta=0)

    :returns:
    A_array: (np array) output A values
    B_array: (np array) output B values
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

    :params:
    x1: (float) first initial A guess
    x2: (float) second initial A guess
    fx1: (float) A end based off of x1
    fx2: (float) B end based off of x2
    tol: (float) tolerance for checking for convergence

    :returns:
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

def data_out(save_text, data, input_name):
    
    # write and save text file
    current_date = datetime.now().strftime("%Y-%m-%d")
    global_params = f"""Data outputs for ultrarelativistic Klein Gordon/Metric solver
    
Max zeta: {ZETA_MAX}
Zeta intervals: {NUM_ZETA_INTERVALS}
Epsilon convergence tolerance: {TOLERANCE}
Write data: {current_date}
    """
    col_label = 'ZETA_S   epsilon   A_0       E/m'
    file_header = global_params + '\n\n' + col_label
    if input_name:
        file_name = input("\nEnter file name for output data (with .txt): ")
    else:
        file_name = 'output.txt'
    if save_text:
        file_data = np.column_stack((ZETA_S_VALS, data['epsilon'], data['A_start'], data['E/m']))
        np.savetxt(file_name, file_data, header=file_header, comments="", fmt='%.4f   %.4f   %.4f   %.4f')
    
    # create and show plots
    fig_idx = 1
    for key, plot_val in SAVE_PARAMS.items():
        if plot_val:
            if key=='AB':
                plt.figure(fig_idx)
                plt.plot(ZETA_VALS, data['A_values'], label="A values", alpha=0.5, marker='.')
                plt.plot(ZETA_VALS, data['B_values'], label="B values", alpha=0.5, marker='.')
                plt.ylabel("A and B")
                plt.xlim(0, 20)
                plt.xlabel("$\zeta$")
                plt.legend()
                plt.grid(True)
                fig_idx+=1
            elif key=='u_bar' or key=='R_tilde':
                plt.figure(fig_idx)
                plt.plot(ZETA_VALS, data[key], label=key, alpha=0.5, marker='.')
                plt.ylabel(key)
                plt.xlim(0, 20)
                plt.xlabel("$\zeta$")
                plt.legend()
                plt.grid(True)
                fig_idx+=1
            elif key=='A_start' or key=='epsilon' or key=='E/m':
                # plots dependent on
                plt.figure(fig_idx)
                plt.plot(ZETA_S_VALS, data[key], label=key, alpha=0.5, marker='.')
                plt.ylabel(key)
                plt.xlabel("$\zeta_s$")
                plt.legend()
                plt.grid(True)
                fig_idx+=1



# Main function that brings it together
def main():
    data = {
        'u_bar': np.zeros(N_MAX),
        'A_values': np.zeros(N_MAX),
        'B_values': np.zeros(N_MAX),
        'R_tilde': np.zeros(N_MAX),
        'epsilon': np.zeros(len(ZETA_S_VALS)),
        'A_start': np.zeros(len(ZETA_S_VALS)),
        'E/m': np.zeros(len(ZETA_S_VALS))
    }

    for j, zeta_s in enumerate(ZETA_S_VALS):
        if j == 0:
            #A_array, B_array, h_tilde = find_fixed_metric(15, zeta_s)
            A_array = np.zeros(N_MAX)
            B_array = np.zeros(N_MAX)
            h_tilde = ZETA_VALS*np.sqrt(np.sqrt(np.exp(2*A_array)/np.exp(2*B_array)))

        A_start = A_array[0]
        epsilon = -1 # initial guess
        AB_sum = 1

        #iterate through equations until convergence
        for i in range(MAX_ITERATIONS):
            print(f"\n\n----- In iter. num. {i+1}, zeta_s={zeta_s}:\n")
            prev_epsilon = epsilon # used to check for epsilon convergence
            
            # Loop through Klein Gordon and Metric equations
            epsilon, u_bar_array = kg_find_epsilon_u(A_array, B_array, h_tilde, zeta_s)
            
            meets_tol = False
            A_start1 = A_start + 0.05
            A_start2 = A_start - 0.05
            AB_adjustments = 0
            tol = 1e-8
            A_array1 = A_array
            B_array1 = B_array
            A_array2 = A_array
            B_array2 = B_array
            while meets_tol == False and AB_adjustments < 50:
                AB_adjustments+=1
                
                # do RK for two different A_0's
                h_tilde1 = ZETA_VALS*np.sqrt(np.sqrt(np.exp(2*A_array1)/np.exp(2*B_array1)))
                R_tilde2_1, dR_tilde2_1, u_tilde = gr_find_Rtilde2_dRtilde2(u_bar_array, A_array1, B_array1, h_tilde1)
                A_array1, B_array1 = gr_RK2(epsilon, R_tilde2_1, dR_tilde2_1, A_array1, B_array1, zeta_s, A_start1)
                fx1 = A_array1[N_MAX-1] + B_array1[N_MAX-1]

                h_tilde2 = ZETA_VALS*np.sqrt(np.sqrt(np.exp(2*A_array2)/np.exp(2*B_array2)))
                R_tilde2_2, dR_tilde2_2, u_tilde = gr_find_Rtilde2_dRtilde2(u_bar_array, A_array2, B_array2, h_tilde2)
                A_array2, B_array2 = gr_RK2(epsilon, R_tilde2_2, dR_tilde2_2, A_array2, B_array2, zeta_s, A_start2)
                fx2 = A_array2[N_MAX-1] + B_array2[N_MAX-1]
                
                # use RK results to do secant method for better A_0 guess
                A_start1, A_start2, meets_tol = find_AB_root(A_start1, A_start2, fx1, fx2, tol)
                #print(f"        New A_start guess: {A_start2}")
            A_array = A_array2
            B_array = B_array2
            A_start = A_start2
            print(f"Runge kutta done, took {AB_adjustments} adjustments")
            
            # recalculate metric elements and h_tilde
            h_tilde[0] = 0
            g_00_array = np.exp(2*A_array)
            g_rr_array = np.exp(2*B_array)
            h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00_array/g_rr_array))

            A_end = A_array[N_MAX-1]
            B_end = B_array[N_MAX-1]

            AB_sum = A_array[N_MAX-1]+B_array[N_MAX-1] # needs to be zero
            print(f"A_end + B_end = {AB_sum}, A_0 = {A_start}")
            print(f"Calculated (lowest) epsilon value: {epsilon}\n")

            # check for epsilon convergence within tolerance
            if abs(epsilon - prev_epsilon) <= TOLERANCE:
                iter_to_tolerance = i+1
                print(f"Tolerance met! Took {iter_to_tolerance} iteration(s) for this adjustment")
                break
        print(f"\n ----------- Iterations and adjustments finished ------------")
        print(f"In {AB_adjustments} adjustments (zeta_s={zeta_s}) the calculated epsilon is {epsilon}, accurate to {TOLERANCE}")
        print(f"A end: {A_end}")
        print(f"B end: {B_end}")
        g_00_array = np.exp(2*A_array)
        g_rr_array = np.exp(2*B_array)

        # store data
        data['u_bar'] = abs(u_bar_array)
        data['R_tilde'] = R_tilde2_2
        data['A_values'] = A_array
        data['B_values'] = B_array
        data['epsilon'][j] = epsilon
        data['A_start'][j] = A_start
        data['E/m'][j] = 1 + (1/2)*zeta_s*epsilon

        '''
        plt.figure(1)
        plt.plot(ZETA_VALS, np.abs(u_bar_array), alpha=0.5, marker='.', label=f"u_bar, $\zeta_s$={zeta_s} , A_end+B_end={AB_sum}")
        plt.xlim(0, 20)
        plt.xlabel("$\zeta$")
        plt.legend()
        plt.grid(True)
        plt.title(f"$\epsilon$={epsilon}\nTol: {TOLERANCE}\nZ_max={ZETA_MAX}, Z_int={NUM_ZETA_INTERVALS}")

        plt.figure(2)
        plt.plot(ZETA_VALS, A_array, alpha=0.5, marker='.', label=f"A, $\zeta_s={zeta_s}$")
        plt.plot(ZETA_VALS, B_array, alpha=0.5, marker='.', label=f"B, , $\zeta_s={zeta_s}$")
        plt.xlim(0, 20)
        plt.xlabel(f"$\zeta$")
        plt.legend()
        plt.grid(True)
        plt.title(f"$\epsilon$={epsilon}\nTol: {TOLERANCE}\nZ_max={ZETA_MAX}, Z_int={NUM_ZETA_INTERVALS}")

        plt.figure(3)
        plt.plot(ZETA_VALS, R_tilde2_2, alpha=0.5, marker='.', label=f"R tilde, $\zeta_s={zeta_s}$")
        plt.legend()
        plt.xlim(0, 20)
        plt.grid(True)
        '''

    _ = data_out(SAVE_PARAMS['write_file'], data, SAVE_PARAMS['custom_name'])
    '''
    A_schw, B_schw, h_tilde_schw = find_fixed_metric(10, 0.5)
    print(f"A_schw end: {A_schw[N_MAX-1]}, B_schw end: {B_schw[N_MAX-1]}")
    plt.figure(2)
    plt.plot(ZETA_VALS, A_schw, color='black', alpha=0.5, linestyle='-', label="A schw")
    plt.plot(ZETA_VALS, B_schw, color='black', alpha=0.5, linestyle='-', label="B schw")
    '''
    plt.title(f"$\zeta_s$={zeta_s}, $\epsilon$={epsilon}\nIter: {iter_to_tolerance}, Tol: {TOLERANCE}\nZ_max={ZETA_MAX}, Z_int={NUM_ZETA_INTERVALS}")
    plt.show()

_ = main()