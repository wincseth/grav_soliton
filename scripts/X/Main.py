#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 08:24:33 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
from Functions import Runge_Kutta, finite_differences, gr_initialize_metric

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

#Main Function

def main():
    
    epsilons=np.zeros_like(ZETA_Sn) #Initializes 0 vector to store final epsilon values
    U_Bars_matrix=np.zeros((len(ZETA), len(ZETA_Sn))) #Stores U_bar values for plotting purposes
    a_initial = np.zeros_like(ZETA_Sn) #Initializes 0 vectors for A and B
    b_initial = np.zeros_like(ZETA_Sn)
    figure1 = plt.figure() #Initializes figure
    ax1 = figure1.add_subplot(111) #Creates big subplot to have common index for both subplots
    ax1_sub1 = figure1.add_subplot(211) #Initializes subplots for A and B
    ax1_sub2 = figure1.add_subplot(212)
    
    for j in range(len(ZETA_Sn)): #Calculates A, B, and epsilon for all ZETA_S values we care about
        ZETA_S = ZETA_Sn[j] #Sets ZETA_S value 
        goo, grr, H_tilde = gr_initialize_metric(ZETA_S) #Initializes A, B, and H_tilde with initial conditions
        for i in range(loops): #Goes through finite differences and Runge Kutta loop to find self satisfying solution
            U_bar, epsilon, e_val = finite_differences(goo, grr, ZETA_S)
            A_array, B_array = Runge_Kutta(U_bar, epsilon, goo, grr, ZETA_S)
            goo = np.exp(2*A_array)
            grr = np.exp(2*B_array)
        U_Bars_matrix[:, j]=U_bar #Saves U_bar
        epsilons[j]=epsilon #Saves epsilon
        ax1_sub1.plot(ZETA, np.exp(2*A_array), label=f'{ZETA_S}', color = (1, j/(len(ZETA_Sn)), j/len(2*ZETA_Sn))) #Plots g_00
        ax1_sub2.plot(ZETA, np.exp(2*B_array), label=f'{ZETA_S}', color = (1, j/(len(ZETA_Sn)), j/len(2*ZETA_Sn))) #Plots g_rr
        a_initial[j] = A_array[0] #Saves initial value of A and B to study
        b_initial[j] = B_array[0]
    print(epsilons)
    print("A initial: ", a_initial)
    print("B initial: ", b_initial)
    figure1.suptitle('g_00 and g_rr')
    ax1.set_xlabel('ZETA')
    
    #Sets outer subplot indeces to invisible
    ax1.spines['top'].set_color('none')
    ax1.spines['bottom'].set_color('none')
    ax1.spines['left'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    #Sets limits to X and Y for subplot, Changes layout, adds legend
    ax1_sub1.set_xlim([0, 20])
    ax1_sub1.set_ylim([0.5, 1.25])
    ax1_sub2.set_xlim([0, 20])
    ax1_sub2.set_ylim([0.5, 1.25])
    ax1_sub1.set_ylabel('g_00')
    ax1_sub2.set_ylabel('g_RR')
    figure1.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig('g_00_and_g_RR.png')

    #Plots U_bars on top of each other to compare
    plt.figure(figsize=(9, 9))
    for i in range(len(ZETA_Sn)):
        plt.plot(ZETA, abs(U_Bars_matrix[:, i]), label=f'ZETA_S = {ZETA_Sn[i]}')
    plt.title('U_Bar for different ZETA_S')
    plt.legend()
    ax = plt.gca()
    ax.set_xlim([0, 25])
    plt.savefig('U_bars.png')
    
    #Plots A and B initial values to study for when we integrate inside out instead of outside in
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, a_initial, label='A')
    plt.plot(ZETA_Sn, b_initial, label='B')
    plt.legend()
    plt.title('Initial values of A and B w.r.t ZETA_S')
    plt.savefig('Initial_A_and_B')


if __name__ == "__main__":
    main()