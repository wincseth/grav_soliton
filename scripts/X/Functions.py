#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:24:17 2024

@author: xaviblast123
"""

#Imported libraries------------------------------------------------------

import numpy as np

# Global Variables-------------------------------------------------------

# User input
n = 500  # Interval Steps
ZETA_MAX = 50
ZETA_Sn = [1]
loops = 33
# -----------------------------------------------------------------------
# Global variables afterwards
DEL = (ZETA_MAX)/(n + 1)  # Spacing of intervals
ZETA = np.arange(0, ZETA_MAX, DEL)  # ZETA array
N_max = len(ZETA)
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
        A: 1D np array
        B; 1D np array
    Outputs:
        goo: 1D np array
        grr: 1D np array
    """

    A = np.log(goo)/2
    B = np.log(grr)/2
    return (A, B)


def matrix_const(goo, grr, ZETA, ZETA_S):
    """
    Creates arrays for matrix values

    Parameters:
        goo: 1D np array
        grr: 1D np array
        ZETA: 1D np array
        ZETA_S: np scalar
    Outputs:
        C: 1D np array
        D: 1D np array
        E: 1D np array
    """
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


def matrix_build(C, D, E):
    """
    Creates matrix for finite differences

    Parameters:
        C: 1D np array
        D: 1D np array
        E: 1D np array
    Outputs:
        Matrix: 2D np array
    """

    matrix = np.zeros((N_max, N_max)) #Initializes zero matrix
    for i in range(N_max): #Places constants in the correct parts of the matrix
        matrix[i, i] = C[i]
        if i > 0: matrix[i, i-1] = D[i]
        if i < N_max-1: matrix[i, i+1] = E[i]

    return matrix


def finite_differences(goo, grr, ZETA_S):
    """
    Uses a finite difference approach to calculate our U_bar array and our epsilon value

    Parameters:
        A: 1D np array
        B: 1D np array
        goo: 1D np array
        grr: 1D np array
    Outputs:
        U_bar: 1D np array
        epsilon: np scalar
    """
    
    A, B = metric(goo, grr)
    C, D, E = matrix_const(goo, grr, ZETA, ZETA_S)
    matrix = matrix_build(C, D, E) #Builds matrix that is dependent on the A and B vectors we got before
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

    return U_bar, epsilon, e_vals[N]


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


def der_of_R(U_bar, H_tilde, goo, grr):
    """
    Derivative of R function

    Parameters:
        R: 1D np array
    Outputs:
        dR: 1D np array
    """

    dR = np.zeros_like(U_bar) #Initializes another zero vector
    U_tilde = np.sqrt(goo/grr)*U_bar #Calculates U_tilde
    for i in range(len(dR)):
        if i == N_max - 1 or i == 0:
            dR[i] = 0 #Derivative is 0 at infinity and 0
            continue
        dR[i] = (U_tilde[i+1] - U_tilde[i-1])/(2*DEL*H_tilde[i]) - U_tilde[i]*(H_tilde[i+1]-H_tilde[i-1])/(2*DEL*H_tilde[i]**2) #Calculates derivative of R using finite differences and product rule
    return dR


def values_for_RK(A, B, R, dR, n, epsilon, ZETA_S):
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
    Outputs:
        dA: np scalar
        dB: np scalar
    """

    ZETA = n*DEL #Calculates ZETA for a specific step
    common = ((ZETA_S**2)*ZETA/8)*(dR**2) + (ZETA_S*ZETA/4)*((1 + ZETA_S*epsilon/2)**2)*(np.exp(2*B-2*A))*(R**2) #Common addition term between dA and dB
    if n == 0: #Sets it equal to 0 for the 0 term
        dA = 0
        dB = 0
        return dA, dB
    dA = (np.exp(2*B) - 1)/(2*ZETA) - ((ZETA_S*ZETA)/4)*np.exp(2*B)*(R**2) + common #Calculates dA
    dB = -(np.exp(2*B) - 1)/(2*ZETA) + ((ZETA_S*ZETA)/4)*np.exp(2*B)*(R**2) + common #Calculates dB
    return dA, dB

def Runge_Kutta(U_bar, epsilon, goo, grr, ZETA_S):

    """
    Performs the Runge Kutta of the second order technique on A and B

    Parameters:
        U_bar: 1D np array
        epsilon: np scalar
        goo: 1D np array
        grr: 1D np array
    Outputs:
        A: 1D np array
        B: 1D np array
    """

    A, B = metric(goo, grr)
    H_tilde = ZETA*np.sqrt(np.sqrt(goo/grr)) #Calculates H_tilde for dR
    R_array=radial_func(U_bar, goo, grr, ZETA) #Calculates Radial function
    dR_array=der_of_R(U_bar, H_tilde, goo, grr) #Calculates derivative of radial function
    dmu_array=np.sqrt(grr/goo)*(np.sqrt(goo/grr)*U_bar)**2 #Calculates derivative of mass function mu
    mu_tilde_end=0
    for n in range(N_max-1):
        mu_tilde_end += DEL*(dmu_array[n]+dmu_array[n+1])/2 #Sums up the whole mass
    
    A[N_max-1]=np.log(1-ZETA_S*mu_tilde_end/ZETA[N_max-1])/2 #Sets end of A vector to include all the mass, our initial condition
    B[N_max-1]=-np.log(1-ZETA_S*mu_tilde_end/ZETA[N_max-1])/2 #Sets end of B vector to include all the mass, our initial condition
    for i in range(N_max-1, 0, -1):
        dA_val, dB_val=values_for_RK(A[i], B[i], R_array[i], dR_array[i], i, epsilon, ZETA_S) #Calculates dA and dB values using our old A and B Vectors
        A_temp=A[i] - DEL * dA_val #Finds temporary values
        B_temp=B[i] - DEL * dB_val
        dA_val2, dB_val2=values_for_RK(A_temp, B_temp, R_array[i-1], dR_array[i-1], i-1, epsilon, ZETA_S) #Calculates dA and dB using our temporary values
        A[i-1]=A[i] - (DEL / 2) * (dA_val + dA_val2) #Sets next step to be the average between the two values
        B[i-1]=B[i] - (DEL / 2) * (dB_val + dB_val2)
    return A, B

def gr_initialize_metric(ZETA_S):
    
    '''
    Initializes our metric equations
    
    Parameters:
        ZETA_S: np scalar
    Outputs:
        a_array: 1D np vector
        b_array: 1D np vector
        h_tilde: 1D np vector
    '''
    g_00_array=np.ones(N_max)
    g_rr_array=np.ones(N_max)
    greater_idx=ZETA > ZETA_S #Finds indeces where ZETA is greater than ZETA_S
    g_00_array[greater_idx]=1 - ZETA_S/ZETA[greater_idx] #Calculates g00's initial condition
    g_rr_array[greater_idx]=1/g_00_array[greater_idx] #Same with grr

    h_tilde=ZETA*np.sqrt(np.sqrt(g_00_array/g_rr_array)) #Calculates h_tilde

    return g_00_array, g_rr_array, h_tilde
    
