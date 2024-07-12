#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:24:17 2024

@author: xaviblast123
"""

#Imported libraries------------------------------------------------------

import numpy as np

# Global Variables-------------------------------------------------------

G = 6.7*10**(-39)  # normalized gravity
M_PL = 1 / np.sqrt(G)  # mass of plank mass
M = 8.2*10**10  # if a equals the atomic Bohr radius
a = 1 / (G*M**3)  # gravitational bohr radius
R_S = 2*G*M  # schwarzschild radius

# Functions--------------------------------------------------------------


def metric(goo, grr):
    """
    Creates the metric arrays using A and B

    Parameters:
        goo: 1D np array
        grr; 1D np array
    Outputs:
        A: 1D np array
        B: 1D np array
    """

    A = np.log(goo)/2
    B = np.log(grr)/2
    return A, B

def back_metric(A, B):
    '''
    Goes from A and B back to the metric
    
    Parameters: 
        A: 1D np array
        B: 1D np array
    Outputs:
        goo: 1D np array
        grr: 1D np array
    '''
    goo = np.exp(2*A)
    grr = np.exp(2*B)
    return goo, grr


def matrix_const(goo, grr, ZETA, ZETA_S, ZETA_MAX):
    """
    Creates arrays for matrix values

    Parameters:
        goo: 1D np array
        grr: 1D np array
        ZETA: 1D np array
        ZETA_S: np scalar
        ZETA_MAX: np scalar
    Outputs:
        C: 1D np array
        D: 1D np array
        E: 1D np array
    """
    N_max = len(ZETA) #Building constants needed
    DEL = ZETA_MAX/len(ZETA)
    
    A, B = metric(goo, grr)
    C = np.zeros_like(ZETA) #Initialize zero vectors for our constants
    D = np.zeros_like(ZETA)
    E = np.zeros_like(ZETA)
    for i in range(N_max):
        if i > 0 and i < N_max-1:
            if i > 0:
                D[i] = -np.sqrt(goo[i]/grr[i]) * np.sqrt(goo[i-1]/grr[i-1])*(1/DEL**2) #Constants in the lower diagonal in the matrix
            if i < N_max-1:
                E[i] = -np.sqrt(goo[i]/grr[i]) * np.sqrt(goo[i+1]/grr[i+1])*(1/DEL**2) #Constants in the upper diagonal in the matrix
            H_ratio = (ZETA[i+1]*np.sqrt(np.sqrt(goo[i+1]/grr[i+1])) - 2*ZETA[i]*np.sqrt(np.sqrt(goo[i]/grr[i])) + ZETA[i-1]*np.sqrt(np.sqrt(goo[i-1]/grr[i-1])))/(ZETA[i]*np.sqrt(np.sqrt(goo[i]/grr[i]))*DEL**2)
        else:
            H_ratio = 0
        C[i] = (goo[i]/grr[i])*H_ratio + (4/ZETA_S) * (np.exp(A[i])*np.sinh(A[i])) + (2/(DEL**2))*(goo[i]/grr[i]) #Constants in the diagonal of the matrix
    return C, D, E


def matrix_build(C, D, E, ZETA):
    """
    Creates matrix for finite differences

    Parameters:
        C: 1D np array
        D: 1D np array
        E: 1D np array
        ZETA: 1D np array
    Outputs:
        Matrix: 2D np array
    """
    N_max = len(ZETA)

    matrix = np.zeros((N_max, N_max)) #Initializes zero matrix
    for i in range(N_max): #Places constants in the correct parts of the matrix
        matrix[i, i] = C[i]
        if i > 0: matrix[i, i-1] = D[i]
        if i < N_max-1: matrix[i, i+1] = E[i]

    return matrix


def finite_differences(goo, grr, ZETA_S, ZETA, ZETA_MAX):
    """
    Uses a finite difference approach to calculate our U_bar array and our epsilon value

    Parameters:
        goo: 1D np array
        grr: 1D np array
        ZETA_S: np scalar
        ZETA: 1D np array
        ZETA_MAX: np scalar
    Outputs:
        U_bar: 1D np array
        epsilon: np scalar
    """
    DEL = (ZETA_MAX)/(len(ZETA))  # Spacing of intervals
    N_max = len(ZETA)
    
    A, B = metric(goo, grr)
    C, D, E = matrix_const(goo, grr, ZETA, ZETA_S, ZETA_MAX)
    matrix = matrix_build(C, D, E, ZETA) #Builds matrix that is dependent on the A and B vectors we got before
    e_vals, e_vecs = np.linalg.eig(matrix) #Grabs eigenvalues and eigenvectors
    epsilons = e_vals/(1+np.sqrt(1+ZETA_S*e_vals/2)) #Calculates the epsilon energies for all the eigenvalues
    N = np.argmin(epsilons) #Finds the index for the smallest epsilon, which is our 0 state energy
    U_bar = e_vecs[:, N] #Gets the eigenvector that corresponds to the 0 energy state
    epsilon = epsilons[N] #Grabs minimum energy

    u_tilde = np.sqrt(goo/grr)*U_bar #Converts into U_tilde
    u_tilde[0] = 0
    u_tilde[N_max-1] = 0
    norm = np.sum(grr*u_tilde**2/np.sqrt(goo)) #Normalizes U_tilde
    u_tilde /= np.sqrt(norm*DEL)

    U_bar = np.sqrt(grr/goo)*u_tilde #Converts back to 

    return U_bar, epsilon


def radial_func(U, goo, grr, ZETA):
    """
    Defining the radial function

    Parameters:
        U: 1D np array
        goo: 1D np array
        grr: 1D np array
        ZETA: 1D np array
    Outputs:
        R: 1D np array
    """

    R = np.zeros_like(U) #Initialize zero vector
    use = np.where(ZETA != 0) #Finds all indeces of ZETA where ZETA isn't 0
    R[use] = np.sqrt(np.sqrt(goo[use]/grr[use])) * U[use] / (ZETA[use]) #Calculates R
    return R


def der_of_R(U_bar, H_tilde, goo, grr, ZETA_MAX):
    """
    Derivative of R function

    Parameters:
        U_bar: 1D np array
        H_tilde: 1D np array
        goo: 1D np array
        grr: 1D np array
        ZETA_MAX: np scalar
    Outputs:
        dR: 1D np array
    """
    N_max = len(U_bar) #Building constants needed
    DEL = ZETA_MAX/N_max
    
    dR = np.zeros_like(U_bar) #Initializes another zero vector
    U_tilde = np.sqrt(goo/grr)*U_bar #Calculates U_tilde
    for i in range(len(dR)):
        if i == N_max - 1 or i == 0:
            dR[i] = 0 #Derivative is 0 at infinity and 0
            continue
        dR[i] = (U_tilde[i+1] - U_tilde[i-1])/(2*DEL*H_tilde[i]) - U_tilde[i]*(H_tilde[i+1]-H_tilde[i-1])/(2*DEL*H_tilde[i]**2) #Calculates derivative of R using finite differences and product rule
    return dR


def values_for_RK(A, B, R, dR, n, epsilon, ZETA_S, ZETA):
    """
    Creates the dA and dB values for RK method

    Parameters:
        A: np scalar
        B: np scalar
        R: np scalar
        dR: np scalar
        n: np scalar
        epsilon; np scalar
        ZETA_S: np scalar
        ZETA: 1D np array
    Outputs:
        dA: np scalar
        dB: np scalar
    """

    common = ((ZETA_S**2)*ZETA[n]/8)*(dR**2) + (ZETA_S*ZETA[n]/4)*((1 + ZETA_S*epsilon/2)**2)*(np.exp(2*B-2*A))*(R**2) #Common addition term between dA and dB
    if n == 0: #Sets it equal to 0 for the 0 term
        dA = 0
        dB = 0
    else:
        dA = (np.exp(2*B) - 1)/(2*ZETA[n]) - ((ZETA_S*ZETA[n])/4)*np.exp(2*B)*(R**2) + common #Calculates dA
        dB = -(np.exp(2*B) - 1)/(2*ZETA[n]) + ((ZETA_S*ZETA[n])/4)*np.exp(2*B)*(R**2) + common #Calculates dB
    return dA, dB

def Runge_Kutta(U_bar, epsilon, goo, grr, ZETA_S, ZETA, ZETA_MAX):

    """
    Performs the Runge Kutta of the second order technique on A and B

    Parameters:
        U_bar: 1D np array
        epsilon: np scalar
        goo: 1D np array
        grr: 1D np array
        ZETA_S: np scalar
        ZETA: 1D np array
        ZETA_MAX: np scalar
    Outputs:
        A: 1D np array
        B: 1D np array
    """
    N_max = len(ZETA)
    DEL = ZETA_MAX/N_max

    A, B = metric(goo, grr)
    H_tilde = ZETA*np.sqrt(np.sqrt(goo/grr)) #Calculates H_tilde for dR
    R_array=radial_func(U_bar, goo, grr, ZETA) #Calculates Radial function
    dR_array=der_of_R(U_bar, H_tilde, goo, grr, ZETA_MAX) #Calculates derivative of radial function
    dmu_array=np.sqrt(grr/goo)*(np.sqrt(goo/grr)*U_bar)**2 #Calculates derivative of mass function mu
    mu_tilde_end=0
    for n in range(N_max-1):
        mu_tilde_end += DEL*(dmu_array[n]+dmu_array[n+1])/2 #Sums up the whole mass
    
    A[N_max-1]=np.log(1-ZETA_S*mu_tilde_end/ZETA[N_max-1])/2 #Sets end of A vector to include all the mass, our initial condition
    B[N_max-1]=-np.log(1-ZETA_S*mu_tilde_end/ZETA[N_max-1])/2 #Sets end of B vector to include all the mass, our initial condition
    for i in range(N_max-1, 0, -1):
        dA_val, dB_val=values_for_RK(A[i], B[i], R_array[i], dR_array[i], i, epsilon, ZETA_S, ZETA) #Calculates dA and dB values using our old A and B Vectors
        A_temp=A[i] - DEL * dA_val #Finds temporary values
        B_temp=B[i] - DEL * dB_val
        dA_val2, dB_val2=values_for_RK(A_temp, B_temp, R_array[i-1], dR_array[i-1], i-1, epsilon, ZETA_S, ZETA) #Calculates dA and dB using our temporary values
        A[i-1]=A[i] - (DEL / 2) * (dA_val + dA_val2) #Sets next step to be the average between the two values
        B[i-1]=B[i] - (DEL / 2) * (dB_val + dB_val2)
    return A, B

def Runge_Kutta_in_out(U_bar, epsilon, A, B, ZETA_S, ZETA, ZETA_MAX, a_start):

    """
    Performs the Runge Kutta of the second order technique on A and B from 0 to ZETA_MAX

    Parameters:
        U_bar: 1D np array
        epsilon: np scalar
        A: 1D np array
        B: 1D np array
        ZETA_S: np scalar
        ZETA: 1D np array
        ZETA_MAX: np scalar
    Outputs:
        A: 1D np array
        B: 1D np array
    """
    A[0] = a_start
    A_new = np.copy(A)
    B_new = np.copy(B)
    N_max = len(ZETA)
    DEL = ZETA_MAX/N_max
    goo, grr = back_metric(A_new, B_new) #Initializes metric
    H_tilde = ZETA*np.sqrt(np.sqrt(goo/grr)) #Calculates H_tilde for dR
    R_array=radial_func(U_bar, goo, grr, ZETA) #Calculates Radial function
    dR_array=der_of_R(U_bar, H_tilde, goo, grr, ZETA_MAX) #Calculates derivative of radial function
    
    for i in range(N_max-1):
        dA_val, dB_val=values_for_RK(A_new[i], B_new[i], R_array[i], dR_array[i], i, epsilon, ZETA_S, ZETA) #Calculates dA and dB values using our old A and B Vectors
        A_temp=A_new[i] + DEL * dA_val #Finds temporary values
        B_temp=B_new[i] + DEL * dB_val
        dA_val2, dB_val2=values_for_RK(A_temp, B_temp, R_array[i+1], dR_array[i+1], i+1, epsilon, ZETA_S, ZETA) #Calculates dA and dB using our temporary values
        A_new[i+1]=A_new[i] + (DEL / 2) * (dA_val + dA_val2) #Sets next step to be the average between the two values
        B_new[i+1]=B_new[i] + (DEL / 2) * (dB_val + dB_val2)
        
    
    return A_new, B_new

def gr_RK2(epsilon, U_bar, A, B, A_start, ZETA, ZETA_S, ZETA_MAX):
    '''
    Uses 2nd order Runge-Kutta ODE method to solve arrays
    for A and B. Returns two numpy arrays, for A and B values respectively.
    '''
    A_array = A
    B_array = B
    A_array[0] = A_start
    B_array[0] = 0
    N_max = len(ZETA)
    DEL = ZETA_MAX/N_max
    goo, grr = back_metric(A_array, B_array)
    H_tilde = ZETA*np.sqrt(np.sqrt(goo/grr))
    Rt_array = radial_func(U_bar, goo, grr, ZETA)
    dRt_array = der_of_R(U_bar, H_tilde, goo, grr, ZETA_MAX)
    
    for n in range(N_max-1):
        A_n = A_array[n]
        B_n = B_array[n]
        Rt_n = Rt_array[n]
        dRt_n = dRt_array[n]
        slope_A_n, slope_B_n = values_for_RK(A_n, B_n, Rt_n, dRt_n, n, epsilon, ZETA_S, ZETA)
        A_temp = A_n + DEL*slope_A_n
        B_temp = B_n + DEL*slope_B_n
        slope_A_temp, slope_B_temp = values_for_RK(A_temp, B_temp, Rt_array[n+1], dRt_array[n+1], n+1, epsilon, ZETA_S, ZETA)
        
        # RK2 method
        A_array[n+1] = A_n + (DEL/2)*(slope_A_n + slope_A_temp)
        B_array[n+1] = B_n + (DEL/2)*(slope_B_n + slope_B_temp)
    
    return A_array, B_array

def gr_initialize_metric(ZETA_S, ZETA):
    
    '''
    Initializes our metric equations
    
    Parameters:
        ZETA_S: np scalar
        ZETA: 1D np array
    Outputs:
        g_00_array: 1D np vector
        g_rr_array: 1D np vector
        h_tilde: 1D np vector
    '''
    N_max = len(ZETA)
    g_00_array=np.ones(N_max)
    g_rr_array=np.ones(N_max)
    greater_idx=ZETA > ZETA_S #Finds indeces where ZETA is greater than ZETA_S
    g_00_array[greater_idx]=1 - ZETA_S/ZETA[greater_idx] #Calculates g00's initial condition
    g_rr_array[greater_idx]=1/g_00_array[greater_idx] #Same with grr

    h_tilde=ZETA*np.sqrt(np.sqrt(g_00_array/g_rr_array)) #Calculates h_tilde

    return g_00_array, g_rr_array, h_tilde
    
def find_fixed_metric(zeta_0, zeta_s, ZETA):
    N_MAX = len(ZETA)
    
    g_00 = np.ones(N_MAX)
    g_rr = np.ones(N_MAX)
    outside_idx = ZETA >= zeta_0
    inside_idx = ZETA < zeta_0
    #print(f"inside={inside_idx}\noutside={outside_idx}")
    def metric_f(zeta_array):
        return 1 - zeta_s*(zeta_array**2)/(zeta_0**3)
    
    g_00[inside_idx] = (1/4)*(3*np.sqrt(metric_f(zeta_0)) - np.sqrt(metric_f(ZETA[inside_idx])))**2
    g_00[outside_idx] = 1 - zeta_s/(ZETA[outside_idx])
    g_rr[outside_idx] = 1/g_00[outside_idx]
    g_rr[inside_idx] = 1/metric_f(ZETA[inside_idx])
    
    h_tilde = ZETA*np.sqrt(np.sqrt(g_00/g_rr))

    A_out = np.log(g_00)/2
    B_out = np.log(g_rr)/2
    return A_out, B_out, h_tilde

def find_AB_root(x1, x2, fx1, fx2):
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
    if abs(fx1) < abs(fx2):
        x1_new = x1
    else:
        x1_new = x2
    
    x2_new = x2 - fx2*(x2 - x1)/(fx2 - fx1)
    #print("Sums: ",fx1, fx2)
    
    return x1_new, x2_new

def integrating_inside_out(a_array, b_array, ZETA, ZETA_S, ZETA_MAX, prev_g1, prev_g2, zeta_0):
    
    
    Rounds = 0
    N_max = len(ZETA)
    error2 = 1
    while error2 > 10**-6:
        Rounds += 1
        goo, grr = back_metric(a_array, b_array)
        
        U_bar, epsilon = finite_differences(goo, grr, ZETA_S, ZETA, ZETA_MAX)
        error = 1
        guess_1 = prev_g1
        guess_2 = prev_g2
        A_array1 = np.copy(a_array)
        B_array1 = np.copy(b_array) 
        A_array2 = np.copy(a_array)
        B_array2 = np.copy(b_array)
        rounds = 0
        while error > 10**-6:
            rounds += 1
            prev_g1 = guess_1
            prev_g2 = guess_2
            A_array1, B_array1 = gr_RK2(epsilon, U_bar, A_array1, B_array1, guess_1, ZETA, ZETA_S, ZETA_MAX)
            fx1 = A_array1[N_max-1] + B_array1[N_max-1]
            A_array2, B_array2 = gr_RK2(epsilon, U_bar, A_array2, B_array2, guess_2, ZETA, ZETA_S, ZETA_MAX)
            fx2 = A_array2[N_max-1] + B_array2[N_max-1]
            guess_1, guess_2 = find_AB_root(guess_1, guess_2, fx1, fx2)
            error = abs(A_array2[N_max-1] + B_array2[N_max-1])
            if np.isnan(guess_1) or np.isnan(guess_2):
                if np.isnan(guess_1):
                    guess_1 = prev_g1
                if np.isnan(guess_2):
                    guess_2 = prev_g2
        if np.isnan(np.sum(A_array1)) or np.isnan(np.sum(A_array2)):
            print("ERRoR ERROR")
            return U_bar, epsilon, a_array, b_array, Rounds, False
        prev_a0 = a_array[0]
        a_array = A_array2
        schwartz_a = np.zeros_like(ZETA)
        index = ZETA > zeta_0
        schwartz_a[index] = np.log(1-(ZETA_S/ZETA[index]))/2
        b_array = B_array2
        print("Rounds: ", rounds, "Current A[0]: ", a_array[0])
        print("Epsilon: ", epsilon, " for ZETA_S = ", ZETA_S)
        error2 = abs(prev_a0 - a_array[0])
        
    return U_bar, epsilon, a_array, b_array, Rounds, True