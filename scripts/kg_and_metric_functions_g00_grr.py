#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Imported libraries------------------------------------------------------
import numpy as np
import sys
# Global Variables-------------------------------------------------------

G = 6.7*10**(-39)  # normalized gravity
M_PL = 1 / np.sqrt(G)  # mass of plank mass
M = 8.2*10**10  # if a equals the atomic Bohr radius
a = 1 / (G*M**3)  # gravitational bohr radius
R_S = 2*G*M  # schwarzschild radius

# Functions--------------------------------------------------------------


def metric_to_AB(g00, grr):
    """
    Creates the metric arrays using A and B

    Parameters:
        g00: 1D np array
        grr; 1D np array
    Outputs:
        A: 1D np array
        B: 1D np array
    """

    A = np.log(g00)/2
    B = np.log(grr)/2
    return A, B

def AB_to_metric(A, B):
    '''
    Goes from A and B back to the metric
    
    Parameters: 
        A: 1D np array
        B: 1D np array
    Outputs:
        g00: 1D np array
        grr: 1D np array
    '''
    g00 = np.exp(2*A)
    grr = np.exp(2*B)
    return g00, grr

def kg_find_matrix_const(g00, grr, zeta_vals, zeta_s, zeta_max):
    """
    Creates arrays for matrix values in the tridiagonal
    matrix used in the Klein Gordon equation.

    Parameters:
        g00: 1D np array
        grr: 1D np array
        zeta_vals: 1D np array
        zeta_s: np scalar
        zeta_max: np scalar
    Outputs:
        C: 1D np array
        D: 1D np array
        E: 1D np array
    """
    N_max = len(zeta_vals) #Building constants needed
    DEL = zeta_max/len(zeta_vals)
    
    A, B = metric_to_AB(g00, grr)
    C = np.zeros_like(zeta_vals) #Initialize zero vectors for our constants
    D = np.zeros_like(zeta_vals)
    E = np.zeros_like(zeta_vals)
    for i in range(N_max):
        if i > 0 and i < N_max-1:
            if i > 0:
                D[i] = -np.sqrt(g00[i]/grr[i]) * np.sqrt(g00[i-1]/grr[i-1])*(1/DEL**2) #Constants in the lower diagonal in the matrix
            if i < N_max-1:
                E[i] = -np.sqrt(g00[i]/grr[i]) * np.sqrt(g00[i+1]/grr[i+1])*(1/DEL**2) #Constants in the upper diagonal in the matrix
            H_ratio = (zeta_vals[i+1]*np.sqrt(np.sqrt(g00[i+1]/grr[i+1])) - 2*zeta_vals[i]*np.sqrt(np.sqrt(g00[i]/grr[i])) + zeta_vals[i-1]*np.sqrt(np.sqrt(g00[i-1]/grr[i-1])))/(zeta_vals[i]*np.sqrt(np.sqrt(g00[i]/grr[i]))*DEL**2)
        else:
            H_ratio = 0
        C[i] = (g00[i]/grr[i])*H_ratio + (4/zeta_s) * (np.exp(A[i])*np.sinh(A[i])) + (2/(DEL**2))*(g00[i]/grr[i]) #Constants in the diagonal of the matrix
    return C, D, E

def kg_build_matrix(C, D, E, zeta_vals):
    """
    Creates a 2d matrix of tridiagonal elements from kg_find_matrix_const(),
    to be used in the Klein Gordon equation for epsilon and u.

    Parameters:
        C: 1D np array
        D: 1D np array
        E: 1D np array
        zeta_vals: 1D np array
    Outputs:
        Matrix: 2D np array
    """
    N_max = len(zeta_vals)

    matrix = np.zeros((N_max, N_max)) #Initializes zero matrix
    for i in range(N_max): #Places constants in the correct parts of the matrix
        matrix[i, i] = C[i]
        if i > 0: matrix[i, i-1] = D[i]
        if i < N_max-1: matrix[i, i+1] = E[i]

    return matrix

def kg_solver(g00, grr, zeta_s, zeta_vals, zeta_max):
    """
    Uses a finite difference approach to calculate our u_bar array and our epsilon value

    Parameters:
        g00: 1D np array
        grr: 1D np array
        zeta_s: np scalar
        zeta_vals: 1D np array
        zeta_max: np scalar
    Outputs:
        u_bar: 1D np array
        epsilon: np scalar
    """
    DEL = (zeta_max)/(len(zeta_vals))  # Spacing of intervals
    N_max = len(zeta_vals)
    
    A, B = metric_to_AB(g00, grr)
    C, D, E = kg_find_matrix_const(g00, grr, zeta_vals, zeta_s, zeta_max)
    matrix = kg_build_matrix(C, D, E, zeta_vals) #Builds matrix that is dependent on the A and B vectors we got before
    e_vals, e_vecs = np.linalg.eig(matrix) #Grabs eigenvalues and eigenvectors
    epsilons = e_vals/(1+np.sqrt(1+zeta_s*e_vals/2)) #Calculates the epsilon energies for all the eigenvalues
    N = np.argmin(epsilons) #Finds the index for the smallest epsilon, which is our 0 state energy
    u_bar = e_vecs[:, N] #Gets the eigenvector that corresponds to the 0 energy state
    epsilon = epsilons[N] #Grabs minimum energy

    u_tilde = np.sqrt(g00/grr)*u_bar #Converts into U_tilde
    u_tilde[0] = 0
    u_tilde[N_max-1] = 0
    norm = np.sum(grr*u_tilde**2/np.sqrt(g00)) #Normalizes U_tilde
    u_tilde /= np.sqrt(norm*DEL)

    u_bar = np.sqrt(grr/g00)*u_tilde #Converts back to 

    return u_bar, epsilon

def metric_find_R_tilde(U, g00, grr, zeta_vals):
    """
    Defining the rescaled radial function, to be used in the general
    relativistic differential equations to redefine the metric.

    Parameters:
        U: 1D np array
        A: 1D np array
        B: 1D np array
        zeta_vals: 1D np array
    Outputs:
        R: 1D np array
    """

    R = np.zeros_like(U) #Initialize zero vector
    use = np.where(zeta_vals != 0) #Finds all indeces of zeta_vals where zeta_vals isn't 0
    R[use] = np.sqrt(np.sqrt(g00[use]/grr[use])) * U[use] / (zeta_vals[use])
    R[0] = R[1] # ensure R tilde at zero is nonzero
    return R

def metric_find_dR_tilde(u_bar, H_tilde, g00, grr, zeta_max):
    """
    Derivative of R tilde function, to be used in the general
    relativistic differential equations to redefine the metric. 

    Parameters:
        u_bar: 1D np array
        H_tilde: 1D np array
        A: 1D np array
        B: 1D np array
        zeta_max: np scalar
    Outputs:
        dR: 1D np array
    """
    N_max = len(u_bar) #Building constants needed
    DEL = zeta_max/N_max
    
    dR = np.zeros_like(u_bar) #Initializes another zero vector
    U_tilde = np.sqrt(g00/grr)*u_bar #Calculates U_tilde
    for i in range(len(dR)):
        if i == N_max - 1 or i == 0:
            dR[i] = 0 #Derivative is 0 at infinity and 0
            continue
        dR[i] = (U_tilde[i+1] - U_tilde[i-1])/(2*DEL*H_tilde[i]) - U_tilde[i]*(H_tilde[i+1]-H_tilde[i-1])/(2*DEL*H_tilde[i]**2) #Calculates derivative of R using finite differences and product rule
    dR[0] = dR[1] # ensure dR tilde at zero is nonzero
    return dR

def metric_find_dA_dB(A, B, R, dR, n, epsilon, zeta_s, zeta_vals):
    """
    Creates the dA and dB values for 2nd order Runge Kutta method,
    to be used in solving the metric differential equations.

    Parameters:
        A: np scalar
        B: np scalar
        R: np scalar
        dR: np scalar
        n: np scalar
        epsilon; np scalar
        zeta_s: np scalar
        zeta_vals: 1D np array
    Outputs:
        dA: np scalar
        dB: np scalar
    """

    common = ((zeta_s**2)*zeta_vals[n]/8)*(dR**2) + (zeta_s*zeta_vals[n]/4)*((1 + zeta_s*epsilon/2)**2)*(np.exp(2*B-2*A))*(R**2) #Common addition term between dA and dB
    if n == 0: #Sets it equal to 0 for the 0 term
        dA = 0
        dB = 0
    else:
        dA = (np.exp(2*B) - 1)/(2*zeta_vals[n]) - ((zeta_s*zeta_vals[n])/4)*np.exp(2*B)*(R**2) + common #Calculates dA
        dB = -(np.exp(2*B) - 1)/(2*zeta_vals[n]) + ((zeta_s*zeta_vals[n])/4)*np.exp(2*B)*(R**2) + common #Calculates dB
    return dA, dB

def metric_RK2(epsilon, u_bar, A, B, A_start, zeta_vals, zeta_s, zeta_max):
    '''
    Uses 2nd order Runge-Kutta ODE method to solve arrays
    for A and B. Returns two numpy arrays, for A and B values respectively.

    params:
        epsilon: (float) scaled binding energy
        u_bar: (np array) modified radial wave function values corresponding to zeta_vals
        A: (np array) starting exponential g00 factors corresponding to zeta_vals
        B: (np array) starting exponential grr factors corresponding to zeta_vals
        A_start: (float) starting guess for A, initial condition
        zeta_vals: (np array) linear zeta array to calculate with
        zeta_s: (float) relativistic parameter
        zeta_max: (float) how far to do zeta calculations to

    returns: 
        A_array: (np array) exponential g00 factors after RK
        B_array: (np array) exponential grr factors after RK
        R_tilde_out: (np array) resulting R_tilde values after RK
    '''
    A_array = A
    B_array = B
    A_array[0] = A_start
    B_array[0] = 0
    N_max = len(zeta_vals)
    DEL = zeta_max/N_max
    g00, grr = AB_to_metric(A_array, B_array)
    H_tilde = zeta_vals*np.sqrt(np.sqrt(g00/grr))
    Rt_array = metric_find_R_tilde(u_bar, g00, grr, zeta_vals)
    dRt_array = metric_find_dR_tilde(u_bar, H_tilde, g00, grr, zeta_max)
    
    for n in range(N_max-1):
        A_n = A_array[n]
        B_n = B_array[n]
        Rt_n = Rt_array[n]
        dRt_n = dRt_array[n]
        slope_A_n, slope_B_n = metric_find_dA_dB(A_n, B_n, Rt_n, dRt_n, n, epsilon, zeta_s, zeta_vals)
        A_temp = A_n + DEL*slope_A_n
        B_temp = B_n + DEL*slope_B_n
        slope_A_temp, slope_B_temp = metric_find_dA_dB(A_temp, B_temp, Rt_array[n+1], dRt_array[n+1], n+1, epsilon, zeta_s, zeta_vals)
        
        # RK2 method
        A_array[n+1] = A_n + (DEL/2)*(slope_A_n + slope_A_temp)
        B_array[n+1] = B_n + (DEL/2)*(slope_B_n + slope_B_temp)
    
    # recalculate R_tilde
    #g00_out, grr_out = AB_to_metric(A_array, B_array)
    R_tilde_out = metric_find_R_tilde(u_bar, A_array, B_array, zeta_vals)

    return A_array, B_array, R_tilde_out
    
def find_fixed_metric(zeta_0, zeta_s, zeta_vals):
    '''
    Uses metric definition of solid mass boundary from https://arxiv.org/abs/1711.00735
    to create an initial guess for the metric, for low zeta_s values.

    params:
        zeta_0: (float) transition zeta where mass ends and metric def. changes
        zeta_s: (float) relativistic parameter
        zeta_vals: (np array) linear zeta array to calculate with

    returns: 
        A_out: (np array) exponential g00 factors corresponding to zeta_vals
        B_out: (np array) exponential grr factors corresponding to zeta_vals
    '''

    N_MAX = len(zeta_vals)
    
    g_00 = np.ones(N_MAX)
    g_rr = np.ones(N_MAX)
    outside_idx = zeta_vals >= zeta_0
    inside_idx = zeta_vals < zeta_0
    #print(f"inside={inside_idx}\noutside={outside_idx}")
    def metric_f(zeta_vals_array):
        return 1 - zeta_s*(zeta_vals_array**2)/(zeta_0**3)
    
    g_00[inside_idx] = (1/4)*(3*np.sqrt(metric_f(zeta_0)) - np.sqrt(metric_f(zeta_vals[inside_idx])))**2
    g_00[outside_idx] = 1 - zeta_s/(zeta_vals[outside_idx])
    g_rr[outside_idx] = 1/g_00[outside_idx]
    g_rr[inside_idx] = 1/metric_f(zeta_vals[inside_idx])


    A_out = np.log(g_00)/2
    B_out = np.log(g_rr)/2
    return A_out, B_out

def metric_find_AB_root(x1, x2, fx1, fx2, tol):
    '''
    Uses secant method to find a new guess for x, to be used
    with RK2 in finding an initial condition for A.

    params:
    x1: (float) first initial A guess
    x2: (float) second initial A guess
    fx1: (float) A end based off of x1
    fx2: (float) B end based off of x2

    returns:
    x1_new: (float) either x1 or x2, whichever has smaller abs(f)
    x2_new: (float) zero intercept and new guess
    meets_tol: (bool) whether or not tolerance is met
    '''
    meets_tol = False

    #slope = (fx2 - fx1)/(x2 - x1)
    if fx1*fx2 > 0: # if outputs are the same sign
        if fx2 < 0:
            x1_new = x1 + 0.05
            print(f"Bad bracketing in A0 secant method, trying again with x1={x1_new}")
            return x1_new, x2, meets_tol
        if fx2 > 0:
            x2_new = x2 - 0.05
            print(f"Bad bracketing in A0 secant method, trying again with x2={x2_new}...")
            return x1, x2_new, meets_tol
    
    if abs(fx1) < abs(fx2):
        x1_new = x1
    else:
        x1_new = x2
    
    x2_new = x2 - fx2*(x2 - x1)/(fx2 - fx1)
    #print("Sums: ",fx1, fx2)
    
    if abs(x2_new - x1_new) <= tol*(abs(x2_new) +abs(x1_new))/2:
        meets_tol = True

    return x1_new, x2_new, meets_tol

def metric_converge_AB(A0_approx, epsilon, u_bar, A, B, zeta_vals, zeta_s, zeta_max, tolerance):
    meets_tol = False
    N_max = len(zeta_vals)
    # (indexing) 0: guess 1 | 1: guess 2 | 2: resulting guess from secant method
    A0 = np.array([A0_approx, A0_approx - 0.1, 0])
    A_arrays = np.column_stack((np.copy(A), np.copy(A), np.copy(A)))
    B_arrays = np.column_stack((np.copy(B), np.copy(B), np.copy(B)))
    fx = np.zeros(3)
    metric_rounds = 0
    while meets_tol == False:
        metric_rounds += 1
        print(f"--- In metric round {metric_rounds}, (zeta_s={zeta_s}):")

        # find fx's
        A_arrays[:, 0], B_arrays[:, 0], R_tilde0 = metric_RK2(epsilon, u_bar, A_arrays[:, 0], B_arrays[:, 0], A0[0], zeta_vals, zeta_s, zeta_max)
        fx[0] = A_arrays[N_max-1, 0] + B_arrays[N_max-1, 0]
        A_arrays[:, 1], B_arrays[:, 1], R_tilde1 = metric_RK2(epsilon, u_bar, A_arrays[:, 1], B_arrays[:, 1], A0[1], zeta_vals, zeta_s, zeta_max)
        fx[1] = A_arrays[N_max-1, 1] + B_arrays[N_max-1, 1]
        print(f"    After RK: A01: {A0[0]}, A02: {A0[1]}, fx1: {fx[0]}, fx2: {fx[1]}")
        '''
        # adjust first two so index 1 has smaller fx absolute value
        if np.abs(fx[0]) < np.abs(fx[1]):
            fx[0], fx[1] = fx[1], fx[0]
            A0[0], A0[1] = A0[1], A0[0]
            print("\n   *Switched arrays so second item has smaller abs(fx) value\n")
        '''
        # use secant method to find new A guess and corresponding fx
        A0[2] = A0[1] - fx[1]*(A0[1] - A0[0])/(fx[1] - fx[0])
        A_arrays[:, 2], B_arrays[:, 2], R_tilde2 = metric_RK2(epsilon, u_bar, A_arrays[:, 2], B_arrays[:, 2], A0[2], zeta_vals, zeta_s, zeta_max)
        fx[2] = A_arrays[N_max-1, 2] + B_arrays[N_max-1, 2]
        print(f"    Secant method gives: A03: {A0[2]}, fx3: {fx[2]}")

        # check if the secant method gives bad A0
        if np.isnan(np.sum(A_arrays[:, 2])) or np.isnan(np.sum(B_arrays[:, 2])):
            print("\n------ NaN encountered in metric from secant method, retrying with wider interval... \n")
            A0[0] += 0.05
            A0[1] -= 0.1
            continue
        else:
            # check for convergence
            if abs(A0[2] - A0[1]) <= tolerance*(abs(A0[2]) + abs(A0[1]))/2:
                meets_tol = True
                A0_out = A0[2]
                A_array_out = A_arrays[:, 2]
                B_array_out = B_arrays[:, 2]
                R_tilde_out = metric_find_R_tilde(u_bar, A_array_out, B_array_out, zeta_vals)
                print(f"\n*** A0 converge met in {metric_rounds} rounds: A0={A0_out} ***\n")
                continue
            '''
            sort_idx = np.argsort(np.abs(fx))
            fx[0] = fx[sort_idx[0]]
            fx[1] = fx[sort_idx[1]]
            fx[2] = 0
            A0[0] = A0[sort_idx[0]]
            A0[1] = A0[sort_idx[1]]
            A0[2] = 0
            print(f"    New sorted vals: A0: {A0}, fx: {fx}\n")
            '''
            if abs(fx[1]) < abs(fx[0]):
                A0[0] = A0[1]
            A0[1] = A0[2]

    return A_array_out, B_array_out, R_tilde_out

def iterate_kg_and_metric(A, B, zeta_vals, zeta_s, zeta_max, A_0_guess, zeta_0):
    '''
    Main function that iterates between Klein Gordon and metric equations to look for converging
    epsilon values, and records the associated u, A, B, data.

    params:
        A: (np array) exponential g00 factors corresponding to zeta_vals
        B: (np array) exponential grr factors corresponding to zeta_vals
        zeta_vals: (np array) linear zeta array to calculate with
        zeta_s: (float) relativistic parameter
        zeta_max: (float) how far to do zeta calculations to
        A_0_guess: (float) initial guess for A at zeta=0, to be adjusted using root finding methods
        zeta_0: (float) for setting up the initial metric guess, the transition zeta where mass ends and metric def. changes
        
    returns: 
        u_bar: (np array) modified radial wave function values corresponding to zeta_vals
        epsilon: (float) scaled binding energy found after converging
        a_array: (np array) exponential g00 factors after kg/metric iteration
        b_array: (np array) exponential grr factors after kg/metric iteration
        R_tilde: (np array) resulting wave function rescaled, related to U
        eps_rounds: (float) how many iterations between kg/metric for epsilon to converge within tolerance
        working_conv: (boolean) whether or not the iterations return valid values, used for finding critical zeta_s
    '''
    
    eps_rounds = 0
    eps_error = 1
    epsilon = -1
    N_max = len(zeta_vals)
    converge_tol = 10e-6 # same tolerance between kg/gr and root finding for A_0
    a_array = A
    b_array = B
    # loop between Klein Gordon and metric equations while epsilon is not converged
    while eps_error > converge_tol:
        eps_rounds += 1
        prev_epsilon = epsilon

        # find u_bar and eps from Klein Gordon
        g00, grr = AB_to_metric(a_array, b_array)
        u_bar, epsilon = kg_solver(g00, grr, zeta_s, zeta_vals, zeta_max)
        a_array, b_array, R_tilde = metric_converge_AB(a_array[0], epsilon, u_bar, a_array, b_array, zeta_vals, zeta_s, zeta_max, converge_tol)

        print(f"--- For eps_round: {eps_rounds}, zeta_s={zeta_s}")
        print(f"Current A[0]: {a_array[0]},")
        print(f"Epsilon: {epsilon}\n")

        '''
        # initialize and loop through RK method until converging A_0 boundary condition is found
        #A_0_g1 = -0.1 # Xavi's version
        #A_0_g2 = -1
        A_0_g1 = a_array[0]
        A_0_g2 = a_array[0] - 0.1
        A_array1 = np.copy(a_array)
        B_array1 = np.copy(b_array) 
        A_array2 = np.copy(a_array)
        B_array2 = np.copy(b_array)
        R_tilde = np.zeros_like(a_array)
        metric_rounds = 0
        meets_metric_tol = False
        #schw_error = 1
        #while schw_error > converge_tol:
        fx1 = 0
        fx2 = 0
        error = 1
        while meets_metric_tol == False:
            metric_rounds += 1
            print(f"--- In metric round {metric_rounds}, (zeta_s={zeta_s}):")
            prev_A_0_g1 = A_0_g1
            prev_A_0_g2 = A_0_g2
            prev_fx1 = fx1
            prev_fx2 = fx2

            A_array1, B_array1, R_tilde1 = metric_RK2(epsilon, u_bar, A_array1, B_array1, A_0_g1, zeta_vals, zeta_s, zeta_max)
            fx1 = A_array1[N_max-1] + B_array1[N_max-1]
            A_array2, B_array2, R_tilde2 = metric_RK2(epsilon, u_bar, A_array2, B_array2, A_0_g2, zeta_vals, zeta_s, zeta_max)
            fx2 = A_array2[N_max-1] + B_array2[N_max-1]
            print(f"    After RK: A01: {A_0_g1}, A02: {A_0_g2}, fx1: {fx1}, fx2: {fx2}")

            A_0_g1, A_0_g2, meets_metric_tol = metric_find_AB_root(A_0_g1, A_0_g2, fx1, fx2, converge_tol)
            print(f"    After Secant Method: A01: {A_0_g1}, A02: {A_0_g2}, fx1: {fx1}, fx2: {fx2}\n")
            if np.isnan(np.sum(A_array1)):
                print("----- NaN encountered in A1 estimate, trying previous guess...\n")
                A_0_g1 = prev_A_0_g1
                fx1 = prev_fx1
                #continue
            if np.isnan(np.sum(A_array2)):
                print("----- NaN encountered in A2 estimate, trying previous guess...\n")
                A_0_g2 = prev_A_0_g2
                fx2 = prev_fx2

            error = abs(A_array2[N_max-1] + B_array2[N_max-1])
            """
            if np.isnan(A_0_g1):
                A_0_g1 = prev_A_0_g1
                #print(f'using previous guess for A0g1, it is now {A_0_g1}')
                print(f"\nNaN found in A_0_g1, using previous one ({A_0_g1})...\n")
                #return u_bar, epsilon, a_array, b_array, R_tilde, eps_rounds, False
                #sys.exit(1)
                #schw_error = 1
            if np.isnan(A_0_g2):
                A_0_g2 = prev_A_0_g2
                #print(f'using previous guess for A0g2, it is now {A_0_g2}')
                print(f"\nNaN found in A_0_g2, using previous one ({A_0_g2})...\n")
                #return u_bar, epsilon, a_array, b_array, R_tilde, eps_rounds, False
                #sys.exit(1)
                #schw_error = 1
            """
        # check for faulty calculations for metric, exit function if found (useful for finding critical zeta_s)
        if np.isnan(np.sum(A_array1)) or np.isnan(np.sum(A_array2)):
            print(f"\n--- ERROR: NaN encountered in metric data, ending epsilon calculation for zeta_s={zeta_s} --- \n")
            return u_bar, epsilon, a_array, b_array, R_tilde, eps_rounds, False
        prev_a0 = a_array[0]
        
        a_array = A_array2
        b_array = B_array2
        R_tilde = R_tilde2
        schwartz_a = np.zeros_like(zeta_vals)
        index = zeta_vals > zeta_0
        schwartz_a[index] = np.log(1-(zeta_s/zeta_vals[index]))/2
        '''

        #eps_error = abs(prev_a0 - a_array[0])
        eps_error = abs(prev_epsilon - epsilon)
        
    return u_bar, epsilon, a_array, b_array, R_tilde, eps_rounds, True
