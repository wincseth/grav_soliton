import numpy as np
import matplotlib.pyplot as plt
import fortran_methods as fm


NUM_ZETA_INTERVALS = 2000 # number of zeta intervals, length of the n arrays - 1
#ZETA_S_VALS = [0.01, 0.1, 0.2, 0.5, 1]
ZETA_MAX = 100
ZETA_0 = 0.5 # semi free parameter corresponding to mass cutoff radius
DELTA = ZETA_MAX/(NUM_ZETA_INTERVALS + 1)
ZETA_VALS = np.arange(0, ZETA_MAX, DELTA)
N_MAX = len(ZETA_VALS)
MAX_ITERATIONS = 40 # how many times to run through the equations
TOLERANCE = 1.0e-6 #level of accuracy for epsilon convergence

G_GRAV = 6.7e-39
M_MASS = 8.2e10
A_BOHR = 1/(G_GRAV*M_MASS**3)
ZETA_S = 0.1
# r = zeta * a

def find_fixed_metric(zeta_0):
    g_00 = np.ones(N_MAX)
    g_rr = np.ones(N_MAX)
    outside_idx = ZETA_VALS >= zeta_0
    inside_idx = ZETA_VALS < zeta_0
    print(f"inside={inside_idx}\noutside={outside_idx}")
    def metric_f(zeta_array):
        return 1 - ZETA_S*(zeta_array**2)/(zeta_0**3)
    
    g_00[inside_idx] = (1/4)*(3*np.sqrt(metric_f(zeta_0)) - np.sqrt(metric_f(ZETA_VALS[inside_idx])))**2
    g_00[outside_idx] = 1 - ZETA_S/(ZETA_VALS[outside_idx])
    g_rr[outside_idx] = 1/g_00[outside_idx]
    g_rr[inside_idx] = 1/metric_f(ZETA_VALS[inside_idx])
    
    print(f"g_00={g_00}\ng_rr={g_rr}")
    return g_00, g_rr

# Klein Gordon equation solver ----------------------------------------------------
# DIFFS: h_tilde_frac and multiplying epsilon def by 2
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
            #h_tilde_frac = (ZETA_VALS[n+1]*np.sqrt(np.sqrt(g_frac_next)) - 2*ZETA_VALS[n]*np.sqrt(np.sqrt(g_frac)) + ZETA_VALS[n-1]*np.sqrt(np.sqrt(g_frac_prev)))/(h_tilde[n]*DELTA**2)
            h_tilde_frac = -zeta_s**2/(4*(zeta_n**2)*(zeta_n**2 - zeta_s**2))
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
    epsilon = 2*lambda_min/(1 + np.sqrt(1 + zeta_s*lambda_min/2))
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


def main():
    g_00, g_rr = find_fixed_metric(ZETA_0)
    A_array = np.log(g_00)/2
    B_array = np.log(g_rr)/2    
    plt.plot(ZETA_VALS, A_array, alpha=0.5, color='blue', label="A directly from metric", marker='.')
    plt.plot(ZETA_VALS, B_array, alpha=0.5, color='red', label="B", marker='.')
    h_tilde = ZETA_VALS*np.sqrt(np.sqrt(g_00/g_rr))

    epsilon, u_bar, hf_out = fm.kg_find_epsilon_u(A_array, B_array, h_tilde, ZETA_S)

    print(f"epsilon={epsilon}")
    plt.xlim(0, 20)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()