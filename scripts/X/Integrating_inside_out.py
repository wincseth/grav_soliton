#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:57:19 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
from Functions import finite_differences, back_metric, find_AB_root, find_fixed_metric
from Functions import gr_RK2 as R_K_in
import pandas as pd
# Global Variables-------------------------------------------------------

# User input
n = 500  # Interval Steps
ZETA_MAX = 50
ZETA_Sn = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.74]
# -----------------------------------------------------------------------
# Global variables afterwards
zeta_0 = 20
DEL = (ZETA_MAX)/(n + 1)  # Spacing of intervals
ZETA = np.arange(0, ZETA_MAX, DEL)  # ZETA array
N_max = len(ZETA)
G = 6.7*10**(-39)  # normalized gravity
M_PL = 1 / np.sqrt(G)  # mass of plank mass
M = 8.2*10**10  # if a equals the atomic Bohr radius
a = 1 / (G*M**3)  # gravitational bohr radius
R_S = 2*G*M  # schwarzschild radius

#Main Function---------------------------------------------------------

def main():
    plt.figure(figsize=(9,9))
    epsilons = []
    A_0s = []
    prev_g1 = -0.1
    prev_g2 = -1
    for i in range(len(ZETA_Sn)):
        ZETA_S = ZETA_Sn[i]
        if i == 0:
            a_array, b_array, h_tilde = find_fixed_metric(zeta_0, ZETA_S, ZETA)
        error2 = 1
        while error2 > 10**-6:
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
                A_array1, B_array1 = R_K_in(epsilon, U_bar, A_array1, B_array1, guess_1, ZETA, ZETA_S, ZETA_MAX)
                fx1 = A_array1[N_max-1] + B_array1[N_max-1]
                A_array2, B_array2 = R_K_in(epsilon, U_bar, A_array2, B_array2, guess_2, ZETA, ZETA_S, ZETA_MAX)
                fx2 = A_array2[N_max-1] + B_array2[N_max-1]
                guess_1, guess_2 = find_AB_root(guess_1, guess_2, fx1, fx2)
                error = abs(A_array2[N_max-1] + B_array2[N_max-1])
                if np.isnan(guess_1) or np.isnan(guess_2):
                    if np.isnan(guess_1):
                        guess_1 = prev_g1
                    if np.isnan(guess_2):
                        guess_2 = prev_g2
            prev_a0 = a_array[0]
            a_array = A_array2
            schwartz_a = np.zeros_like(ZETA)
            index = ZETA > zeta_0
            schwartz_a[index] = np.log(1-(ZETA_S/ZETA[index]))/2
            b_array = B_array2
            print("Rounds: ", rounds, " Loop: ", i, "Current A[0]: ", a_array[0])
            print("Epsilon: ", epsilon, " for ZETA_S = ", ZETA_S)
            error2 = abs(prev_a0 - a_array[0])
            
        epsilons.append(epsilon)
        A_0s.append(a_array[0])
        if i == 0:
            df_U_bar = pd.DataFrame(abs(U_bar), columns = [f'U Bar of {ZETA_S}'], index = ZETA)
            df_A = pd.DataFrame(a_array, columns = [f'A of {ZETA_S}'], index = ZETA)
            df_B = pd.DataFrame(b_array, columns = [f'B of {ZETA_S}'], index = ZETA)
        else:
            df_U_bar[f'U Bar of {ZETA_S}'] = abs(U_bar)
            df_A[f'A of {ZETA_S}'] = a_array
            df_B[f'B of {ZETA_S}'] = b_array
        print(epsilons)
        print("Final A: ", a_array[N_max-1], "Final schwartz: ", schwartz_a[N_max-1])

    df_ZETA_S = pd.DataFrame({'Epsilons' : epsilons,
                              'A_0' : A_0s}, index = ZETA_Sn)
    df_ZETA = pd.concat([df_U_bar, df_A, df_B], axis = 1)
    
    
    df_ZETA.to_csv("Dependent_on_ZETA.csv")
    df_ZETA_S.to_csv("Dependent_on_ZETA_S.csv")
    
    


    
_=main()