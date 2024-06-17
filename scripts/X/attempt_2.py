#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:24:17 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt

# Global Variables-------------------------------------------------------

# User input
n = 500  # Interval Steps
ZETA_MAX = 50
ZETA_Sn = [0.01, 0.1, 0.2, 0.5, 1]
loops = 25
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
    C = np.zeros_like(ZETA)
    D = np.zeros_like(ZETA)
    E = np.zeros_like(ZETA)
    for i in range(N_max):
        if i > 0 and i < N_max-1:
            if i > 0:
                D[i] = -np.sqrt(goo[i]/grr[i]) * np.sqrt(goo[i-1]/grr[i-1])*(1/DEL**2)
            if i < N_max-1:
                E[i] = -np.sqrt(goo[i]/grr[i]) * np.sqrt(goo[i+1]/grr[i+1])*(1/DEL**2)
            H_ratio = (ZETA[i+1]*np.sqrt(np.sqrt(goo[i+1]/grr[i+1])) - 2*ZETA[i]*np.sqrt(np.sqrt(goo[i]/grr[i])) + ZETA[i-1]*np.sqrt(np.sqrt(goo[i-1]/grr[i-1])))/(ZETA[i]*np.sqrt(np.sqrt(goo[i]/grr[i]))*DEL**2)
        else:
            H_ratio = 0
        C[i] = (goo[i]/grr[i])*H_ratio + (4/ZETA_S) * (np.exp(A[i])*np.sinh(A[i])) + (2/(DEL**2))*(goo[i]/grr[i])
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
    epsilons = e_vals/(1+np.sqrt(1+ZETA_S*e_vals/2))
    N = np.argmin(epsilons)
    U_bar = e_vecs[:, N]
    epsilon = epsilons[N]

    u_tilde = np.sqrt(goo/grr)*U_bar
    u_tilde[0] = 0
    u_tilde[N_max-1] = 0
    norm = np.sum(grr*u_tilde**2/np.sqrt(goo))
    u_tilde /= np.sqrt(norm*DEL)

    U_bar = np.sqrt(grr/goo)*u_tilde

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
    R[use] = np.sqrt(np.sqrt(goo[use]/grr[use])) * U[use] / (ZETA[use])
    return R


def der_of_R(U_bar, H_tilde, goo, grr):
    """
    Derivative of R function

    Parameters:
        R: 1D np array
    Outputs:
        dR: 1D np array
    """

    dR = np.zeros_like(U_bar)
    U_tilde = np.sqrt(goo/grr)*U_bar
    for i in range(len(dR)):
        if i == N_max - 1 or i == 0:
            dR[i] = 0
            continue
        dR[i] = (U_tilde[i+1] - U_tilde[i-1])/(2*DEL*H_tilde[i]) - U_tilde[i]*(H_tilde[i+1]-H_tilde[i-1])/(2*DEL*H_tilde[i]**2)
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
    Outputs:
        dA: np scalar
        dB: np scalar
    """

    ZETA = n*DEL
    common = ((ZETA_S**2)*ZETA/8)*(dR**2) + (ZETA_S*ZETA/4)*((1 + ZETA_S*epsilon/2)**2)*(np.exp(2*B-2*A))*(R**2)
    
    if n == 0:
        dA = 0
        dB = 0
        return dA, dB
    dA = (np.exp(2*B) - 1)/(2*ZETA) - ((ZETA_S*ZETA)/4)*np.exp(2*B)*(R**2) + common
    dB = -(np.exp(2*B) - 1)/(2*ZETA) + ((ZETA_S*ZETA)/4)*np.exp(2*B)*(R**2) + common
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

    goo, grr=metric(A, B)
    H_tilde = ZETA*np.sqrt(np.sqrt(goo/grr))
    R_array=radial_func(U_bar, goo, grr, ZETA)
    dR_array=der_of_R(U_bar, H_tilde, goo, grr)
    dmu_array=np.sqrt(grr/goo)*(np.sqrt(goo/grr)*U_bar)**2
    mu_tilde_end=0
    for n in range(N_max-1):
        mu_tilde_end += DEL*(dmu_array[n]+dmu_array[n+1])/2
    
    A[N_max-1]=np.log(1-ZETA_S*mu_tilde_end/ZETA[N_max-1])/2
    B[N_max-1]=-np.log(1-ZETA_S*mu_tilde_end/ZETA[N_max-1])/2
    for i in range(N_max-1, 0, -1):
        dA_val, dB_val=values_for_RK(A[i], B[i], R_array[i], dR_array[i], i, epsilon, ZETA_S)
        A_temp=A[i] - DEL * dA_val
        B_temp=B[i] - DEL * dB_val
        dA_val2, dB_val2=values_for_RK(A_temp, B_temp, R_array[i-1], dR_array[i-1], i-1, epsilon, ZETA_S)
        A[i-1]=A[i] - (DEL / 2) * (dA_val + dA_val2)
        B[i-1]=B[i] - (DEL / 2) * (dB_val + dB_val2)
    return A, B

def gr_initialize_metric(ZETA_S):
    a_array=np.zeros(N_max)
    b_array=np.zeros(N_max)
    g_00_array=np.ones(N_max)
    g_rr_array=np.ones(N_max)
    greater_idx=ZETA > ZETA_S
    a_array[greater_idx]=np.log(1-ZETA_S/ZETA[greater_idx])/2
    b_array[greater_idx]=-np.log(1-ZETA_S/ZETA[greater_idx])/2
    g_00_array[greater_idx]=1 - ZETA_S/ZETA[greater_idx]
    g_rr_array[greater_idx]=1/g_00_array[greater_idx]

    h_tilde=ZETA*np.sqrt(np.sqrt(g_00_array/g_rr_array))

    return a_array, b_array, h_tilde
    

# Main Function------------------------------------------------------------

def main():
    epsilons=np.zeros_like(ZETA_Sn)
    U_Bars_matrix=np.zeros((len(ZETA), len(ZETA_Sn)))
    #goal = [-0.1631, -0.1657, -0.1688, -0.1795, -0.2045]
    color = ['red', 'orange', 'green', 'blue', 'violet']
    a_initial = np.zeros_like(ZETA_Sn)
    b_initial = np.zeros_like(ZETA_Sn)
    figure1 = plt.figure()
    ax1 = figure1.add_subplot(111)
    ax1_sub1 = figure1.add_subplot(211)
    ax1_sub2 = figure1.add_subplot(212)
    for j in range(len(ZETA_Sn)):
        ZETA_S=ZETA_Sn[j]
        A_array, B_array, H_tilde = gr_initialize_metric(ZETA_S)
        for i in range(loops):
            U_bar, epsilon = finite_differences(A_array, B_array, ZETA_S)
            A_array, B_array = Runge_Kutta(U_bar, epsilon, A_array, B_array, ZETA_S)
        U_Bars_matrix[:, j]=U_bar
        epsilons[j]=epsilon
        ax1_sub1.plot(ZETA, np.exp(2*A_array), label=f'{ZETA_S}', color = color[j])
        ax1_sub2.plot(ZETA, np.exp(2*B_array), label=f'{ZETA_S}', color = color[j])
        a_initial[j] = A_array[0]
        b_initial[j] = B_array[0]
    figure1.suptitle('g_00 and g_rr')
    ax1.set_xlabel('ZETA')
    
    ax1.spines['top'].set_color('none')
    ax1.spines['bottom'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    ax1_sub1.set_xlim([0, 20])
    ax1_sub1.set_ylim([0.5, 1.25])
    ax1_sub2.set_xlim([0, 20])
    ax1_sub2.set_ylim([0.5, 1.25])
    ax1_sub1.set_ylabel('g_00')
    ax1_sub2.set_ylabel('g_RR')
    figure1.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.savefig('g_00_and_g_RR.png')

    plt.figure(figsize=(9, 9))
    for i in range(len(ZETA_Sn)):
        plt.plot(ZETA, abs(U_Bars_matrix[:, i]), label=f'ZETA_S = {ZETA_Sn[i]}')
    plt.title('U_Bar for different ZETA_S')
    plt.legend()
    ax = plt.gca()
    ax.set_xlim([0, 25])
    plt.show()
    plt.savefig('U_bars.png')
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, a_initial, label='A')
    plt.plot(ZETA_Sn, b_initial, label='B')
    plt.legend()
    plt.title('Initial values of A and B w.r.t ZETA_S')
    plt.show()
    plt.savefig('Initial_A_and_B')
    
    print(epsilons)
    #print("Percent errors: ", 100*(goal-epsilons)/goal)

_=main()
