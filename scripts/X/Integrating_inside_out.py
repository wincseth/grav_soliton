#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:57:19 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt

# Global Variables-------------------------------------------------------

# User input
n = 750  # Interval Steps
ZETA_MAX = 120
ZETA_S = 1
loops = 50
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

def metric(A, B):
    """
    Creates the metric arrays using A and B

    Parameters:
        A: 1D np array
        B; 1D np array
    Outputs:
        goo: 1D np array
        grr: 1D np array
    """

    goo = np.exp(2*A)
    grr = np.exp(2*B)
    return (goo, grr)

def matrix_const(A, B, ZETA, ZETA_S):
    """
    Creates arrays for matrix values

    Parameters:
        A: 1D np array
        B: 1D np array
        ZETA: 1D np array
        ZETA_S: np scalar
    Outputs:
        C: 1D np array
        D: 1D np array
        E: 1D np array
    """

    goo, grr = metric(A, B)
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


def finite_differences(A, B, ZETA_S):
    """
    Uses a finite difference approach to calculate our U_bar array and our epsilon value

    Parameters:
        A: 1D np array
        B: 1D np array
    Outputs:
        U_bar: 1D np array
        epsilon: np scalar
    """

    goo, grr = metric(A, B)
    C, D, E = matrix_const(A, B, ZETA, ZETA_S)
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

    U_bar = np.sqrt(grr/goo)*u_tilde #Converts back to U_bar

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

def Runge_Kutta(U_bar, epsilon, A, B, ZETA_S):

    """
    Performs the Runge Kutta of the second order technique on A and B

    Parameters:
        U_bar: 1D np array
        epsilon: np scalar
        A: 1D np array
        B: 1D np array
    Outputs:
        A_new: 1D np array
        B_new: 1D np array
    """

    goo, grr = metric(A, B) #Initializes metric
    H_tilde = ZETA*np.sqrt(np.sqrt(goo/grr)) #Calculates H_tilde for dR
    R_array=radial_func(U_bar, goo, grr, ZETA) #Calculates Radial function
    dR_array=der_of_R(U_bar, H_tilde, goo, grr) #Calculates derivative of radial function
    
    
    for i in range(0, N_max-1, 1):
        dA_val, dB_val=values_for_RK(A[i], B[i], R_array[i], dR_array[i], i, epsilon, ZETA_S) #Calculates dA and dB values using our old A and B Vectors
        A_temp=A[i] + DEL * dA_val #Finds temporary values
        B_temp=B[i] + DEL * dB_val
        dA_val2, dB_val2=values_for_RK(A_temp, B_temp, R_array[i+1], dR_array[i+1], i+1, epsilon, ZETA_S) #Calculates dA and dB using our temporary values
        A[i+1]=A[i] + (DEL / 2) * (dA_val + dA_val2) #Sets next step to be the average between the two values
        B[i+1]=B[i] + (DEL / 2) * (dB_val + dB_val2)
    return A, B

#Main Function---------------------------------------------------------

def main():
    a_array = np.zeros_like(ZETA)
    b_array = np.zeros_like(ZETA)
    a_array[0] = 0.28117229
    b_array[0] = -0.5146
    for i in range(loops):
        U_bar, epsilon = finite_differences(a_array, b_array, ZETA_S)
        Runge_Kutta(U_bar, epsilon, a_array, b_array, ZETA_S)
        print(epsilon)
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA, np.exp(2*a_array), label="A")
    plt.plot(ZETA, np.exp(b_array*2), label ='B')
    plt.legend()
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA, abs(U_bar))
    
_=main()