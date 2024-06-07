#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:24:17 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt

#Global Variables-------------------------------------------------------

#User input
n = 375 #Interval Steps
ZETA_MAX = 190
ZETA_Sn = np.linspace(0.1, 1, 10)
loops = 30
#-----------------------------------------------------------------------
#Global variables afterwards
DEL = (ZETA_MAX)/(n + 1) #Spacing of intervals
ZETA = np.arange(0, ZETA_MAX, DEL) #ZETA array
N_max = len(ZETA)
G = 6.7*10**(-39) #normalized gravity
M_PL = 1 / np.sqrt(G) #mass of plank mass
M = 8.2*10**10 #if a equals the atomic Bohr radius
a = 1 /(G*M**3)#gravitational bohr radius
R_S = 2*G*M #schwarzschild radius

#Functions--------------------------------------------------------------

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
    return(goo, grr)

def matrix_const(A, B, ZETA, ZETA_S):
    
    """
    Creates arrays for matrix values
    
    Parameters:
        A: 1D np array
        B: 1D np array
        Zeta: 1D np array
    Outputs:
        C: 1D np array
        D: 1D np array
        E: 1D np array
    """
    
    goo, grr = metric(A, B)
    H_ratio = ((ZETA_S**2)/(4*(ZETA**2)*((ZETA**2)-(ZETA_S**2))))
    H_ratio[0] = 0
    C = -(goo/grr)*H_ratio + (4/ZETA_S)*(np.exp(A)*np.sinh(A)) + (2/(DEL**2))*(goo/grr)
    D = np.zeros_like(ZETA)
    E = np.zeros_like(ZETA)
    for i in range(N_max):
        if i > 0:
            D[i] = -np.sqrt(goo[i]/grr[i])*np.sqrt(goo[i-1]/grr[i-1])*(1/DEL**2)
        if i < N_max-1:
            E[i] = -np.sqrt(goo[i]/grr[i])*np.sqrt(goo[i+1]/grr[i+1])*(1/DEL**2)
    
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
    
    matrix = np.zeros((N_max, N_max))
    for i in range(N_max):
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
    matrix = matrix_build(C, D, E)
    e_vals, e_vecs = np.linalg.eig(matrix)
    N = np.argmin(abs(e_vals))
    U_bar = e_vecs[:, N]
    norm = sum(np.sqrt(goo)*DEL*U_bar**2)
    U_bar = U_bar / norm  # Correct normalization
    U_bar[0] = 0
    epsilon = e_vals[N]/(1+np.sqrt(1+ZETA_S*e_vals[N]/2))
    
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
    
    R = np.zeros_like(U)
    use = np.where(ZETA != 0)
    R[use] = np.sqrt(np.sqrt(goo[use] / grr[use])) * U[use] / (ZETA[use])
    return R
    
def der_of_R(R):
    
    """
    Derivative of R function
    
    Parameters:
        R: 1D np array
    Outputs: 
        dR: 1D np array
    """
    
    dR = np.zeros(len(R))
    for i in range(len(R)):
        if i == N_max - 1 or i == 0:
            dR[i] = 0
            continue
        dR[i] = (R[i + 1] - R[i - 1]) / (2 * DEL)
    return dR

def values_for_RK(A, B, ZETA, R, dR, n, epsilon, ZETA_S):
    
    """
    Creates the dA and dB values for RK method
    
    Parameters:
        A: np scalar
        B: np scalar
        ZETA: 1D np array
        R: 1D np array
        dR: 1D np array
        n: np scalar
        epsilon; np scalar
    Outputs:
        dA: np scalar
        dB: np scalar
    """
    
    dA = DEL * ((1 / (2 * ZETA[n]) * (np.exp(2 * B) - 1)) - (ZETA_S * ZETA[n] / 4) * np.exp(2 * B) * (R[n]**2) + ((ZETA[n] * ZETA_S**2) / 8) * dR[n]**2 + (ZETA_S * ZETA[n] / 4) * ((1 + .5 * ZETA_S * epsilon)**2) * (np.exp(2 * (B - A))) * R[n]**2)
    dB = DEL * (-(1 / (2 * ZETA[n]) * (np.exp(2 * B) - 1)) + (ZETA_S * ZETA[n] / 4) * np.exp(2 * B) * (R[n]**2) + ((ZETA[n] * ZETA_S**2) / 8) * dR[n]**2 + (ZETA_S * ZETA[n] / 4) * ((1 + .5 * ZETA_S * epsilon)**2) * (np.exp(2 * (B - A))) * R[n]**2)
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
    
    goo, grr = metric(A, B)
    R_array = radial_func(U_bar, goo, grr, ZETA)
    dR_array = der_of_R(R_array)
    A_new = np.zeros_like(U_bar)
    B_new = np.zeros_like(U_bar)
    for i in range(len(A)):
        if i == 0:
            A_new[i] = 0
            B_new[i] = 0
            continue
        if i == 1:
            dA_val = 0
            dB_val = 0
        else:
            dA_val, dB_val = values_for_RK(A_new[i - 1], B_new[i - 1], ZETA, R_array, dR_array, i - 1, epsilon, ZETA_S)
        A_temp = A_new[i - 1] + DEL * dA_val
        B_temp = B_new[i - 1] + DEL * dB_val
        dA_val2, dB_val2 = values_for_RK(A_temp, B_temp, ZETA, R_array, dR_array, i, epsilon, ZETA_S)
        A_new[i] = A_new[i - 1] + (DEL / 2) * (dA_val + dA_val2)
        B_new[i] = B_new[i - 1] + (DEL / 2) * (dB_val + dB_val2)
    return A_new, B_new

#Main Function------------------------------------------------------------

def main():
    plt.figure(figsize=(9, 9))
    for j in range(len(ZETA_Sn)):
        ZETA_S = ZETA_Sn[j]
        A_array = np.zeros_like(ZETA)
        B_array = np.zeros_like(ZETA)
        for i in range(loops):
            U_bar, epsilon = finite_differences(A_array, B_array, ZETA_S)
            A_array, B_array = Runge_Kutta(U_bar, epsilon, A_array, B_array, ZETA_S)
            print(ZETA_S)
            print(epsilon)
        plt.plot(ZETA, abs(U_bar), label=f'zeta_s value of {ZETA_Sn[j]}')
    ax = plt.gca()
    ax.set_xlim([0, 25])
    plt.legend()
    
_=main()