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
n = 200 #Interval Steps
ZETA_MAX = 150
ZETA_S = 0.01
loops = 50
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

def matrix_const(A, B, ZETA):
    
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
    for i in range(N_max-1):
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
    for i in range(N_max-1):
        matrix[i, i] = C[i]
        if i > 0: matrix[i, i-1] = D[i]
        if i < N_max-1: matrix[i, i+1] = E[i]
    return matrix

def finite_differences(A, B):
    
    """
    Uses a finite difference approach to calculate our U_bar array and our epsilon value
    
    Parameters:
        A: 1D np array
        B: 1D np array
    Outputs:
        U_bar: 1D np array
        epsilon: np scalar
    """
    
    goo = metric(A, B)[0]
    C, D, E = matrix_const(A, B, ZETA)
    matrix = matrix_build(C, D, E)
    e_vals, e_vecs = np.linalg.eig(matrix)
    N = np.argmin(e_vals)
    U_bar = e_vecs[:, N]
    U_bar /= np.sqrt(sum(np.sqrt(goo)*U_bar**2*DEL))
    epsilon = e_vals/(1+np.sqrt(1+ZETA_S*e_vals/2))
    return U_bar, epsilon
    
def main():