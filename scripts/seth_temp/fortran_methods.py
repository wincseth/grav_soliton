import numpy as np
import matplotlib.pyplot as plt

# GLOBAL PARAMETERS:
NUM_ZETA_INTERVALS = 800 # number of zeta intervals, length of the n arrays - 1
ZETA_S_VALS = [0.01, 1]
ZETA_MAX = 40
DELTA = ZETA_MAX/(NUM_ZETA_INTERVALS + 1)
ZETA_VALS = np.arange(0, ZETA_MAX, DELTA)
N_MAX = len(ZETA_VALS)
MAX_ITERATIONS = 40 # how many times to run through the equations
TOLERANCE = 1.0e-6 #level of accuracy for epsilon convergence

G_GRAV = 6.7e-39
M_MASS = 8.2e10
A_BOHR = 1/(G_GRAV*M_MASS**3)

PLOT_PARAMS = {
    'u_bar': False,
    'h_tilde': False,
    'AB': True
}

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

    return epsilon, u_bars, hf_out

def gr_initialize_metric(zeta_s):
    a_array = np.zeros(N_MAX)
    b_array = np.zeros(N_MAX)
    g_00_array = np.ones(N_MAX)
    g_rr_array = np.ones(N_MAX)
    greater_idx = ZETA_VALS > zeta_s
    a_array[greater_idx] = np.log(1-zeta_s/ZETA_VALS[greater_idx])/2
    b_array[greater_idx] = -np.log(1-zeta_s/ZETA_VALS[greater_idx])/2
    g_00_array[greater_idx] = 1 - zeta_s/ZETA_VALS[greater_idx]
    g_rr_array[greater_idx] = 1/g_00_array[greater_idx]

    h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00_array/g_rr_array))

    return a_array, b_array, h_tilde


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

def gr_BC(u_tilde, a_array, b_array, zeta_s):
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
    

    A_end = np.log(1-zeta_s*mu_tilde_end/ZETA_VALS[N_MAX-1])/2
    B_end = -A_end
    return A_end, B_end

def gr_RK2(epsilon, Rt_array, dRt_array, u_tilde, A_array, B_array, zeta_s):
    '''
    Uses 2nd order Runge-Kutta ODE method to solve arrays
    for A and B. Returns two numpy arrays, for A and B values respectively.
    '''
    
    A_array[N_MAX-1], B_array[N_MAX-1] = gr_BC(u_tilde, A_array, B_array, zeta_s)
    for n in range(N_MAX-1, 0, -1):
        A_n = A_array[n]
        B_n = B_array[n]
        Rt_n = Rt_array[n]
        dRt_n = dRt_array[n]
        slope_A_n, slope_B_n = gr_find_AB_slope(A_n, B_n, n, epsilon, Rt_n, dRt_n, zeta_s)
        
        A_temp = A_n - DELTA*slope_A_n
        B_temp = B_n - DELTA*slope_B_n
        slope_A_temp, slope_B_temp = gr_find_AB_slope(A_temp, B_temp, n-1, epsilon, Rt_array[n-1], dRt_array[n-1], zeta_s)
        
        # RK2 method
        A_array[n-1] = A_n - (DELTA/2)*(slope_A_n + slope_A_temp)
        B_array[n-1] = B_n - (DELTA/2)*(slope_B_n + slope_B_temp)
    A_array[0] = 0
    B_array[0] = 0
    return A_array, B_array

# Main function that brings it together
def main():

    # key values will be replaced with plot arrays
    data = {
        'u_bar': True,
        'h_tilde': True,
        'A_values': True,
        'B_values': True
    }
    fig_idx = 1
    for j in ZETA_S_VALS:
        zeta_s = j
        A_array, B_array, h_tilde = gr_initialize_metric(zeta_s)
        epsilon = -1 # initial guess
    
        #iterate through equations until convergence
        for i in range(MAX_ITERATIONS):
            print(f"\n\n----- In iteration number {i+1}, zeta_s={zeta_s}:\n")
            
            prev_epsilon = epsilon # used to check for epsilon convergence
            
            # Loop through Klein Gordon and Metric equations
            epsilon, u_bar_array, hf_out = kg_find_epsilon_u(A_array, B_array, h_tilde, zeta_s)
            R_tilde2, dR_tilde2, u_tilde = gr_find_Rtilde2_dRtilde2(u_bar_array, A_array, B_array, h_tilde)
            A_array, B_array = gr_RK2(epsilon, R_tilde2, dR_tilde2, u_tilde, A_array, B_array, zeta_s)
            # recalculate metric elements and h_tilde
            h_tilde[0] = 0
            g_00_array = np.exp(2*A_array)
            g_rr_array = np.exp(2*B_array)
            h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00_array/g_rr_array))

            print(f"Calculated (lowest) epsilon value: {epsilon}\n")
            #print(f"    A vals: {A_array}\nB vals: {B_array}\n")

            # assign values to dictionary
            data['u_bar'] = abs(u_bar_array)
            data['h_tilde'] = h_tilde
            data['A_values'] = A_array
            data['B_values'] = B_array

            # check for epsilon convergence within tolerance
            if abs(epsilon - prev_epsilon) <= TOLERANCE:
                iter_to_tolerance = i+1
                break
        print(f"In {iter_to_tolerance} iterations the calculated epsilon is {epsilon}, accurate to {TOLERANCE}")
            
        g_00_array = np.exp(2*A_array)
        g_rr_array = np.exp(2*B_array)

        # plot using the global dictionary
        for key, plot_val in PLOT_PARAMS.items():
            if plot_val and key=='AB':
                plt.figure(fig_idx)
                plt.plot(ZETA_VALS, data['A_values'], label="A values", alpha=0.5, marker='.')
                plt.plot(ZETA_VALS, data['B_values'], label="B values", alpha=0.5, marker='.')
                plt.ylabel("A and B")

                fig_idx+=1
            elif plot_val:
                plt.figure(fig_idx)
                plt.plot(ZETA_VALS, data[key], label=key, alpha=0.5, marker='.')
                plt.ylabel(key)
                fig_idx+=1
            plt.xlim(0, 20)
            plt.xlabel("$\zeta$")
            plt.legend()
            plt.grid(True)
        plt.title(f"$\zeta_s$={zeta_s}, $\epsilon$={epsilon}\nIter: {iter_to_tolerance}, Tol: {TOLERANCE}\nZ_max={ZETA_MAX}, Z_int={NUM_ZETA_INTERVALS}")
            
        #plt.figure(1)
        #plt.plot(ZETA_VALS, abs(u_bar_array), alpha=0.75, label=f"$\zeta_s={zeta_s}$, $\epsilon={epsilon}$")
        #plt.plot(ZETA_VALS, hf_out, alpha=0.75, label="h tilde fraction", marker='.')
        
        #plt.title(f"$\zeta_s$={zeta_s}, $\epsilon$={epsilon}")

    #plt.xlim(0, 20)
    #plt.legend()
    #plt.grid(True)
    plt.show()

_ = main()