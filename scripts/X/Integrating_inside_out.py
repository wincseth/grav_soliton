#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:57:19 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
from Functions import finite_differences, back_metric, secant_root_alg
from Functions import Runge_Kutta_in_out as R_K_in
# Global Variables-------------------------------------------------------

# User input
n = 500  # Interval Steps
ZETA_MAX = 50
ZETA_Sn = [.2]
loops = 22
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

#Main Function---------------------------------------------------------

def main():
    plt.figure(figsize=(9,9))
    for i in range(len(ZETA_Sn)):
        ZETA_S = ZETA_Sn[i]
        a_array = np.zeros_like(ZETA)
        b_array = np.zeros_like(ZETA)
        index = ZETA > ZETA_S
        a_array[index] = np.log(1-ZETA_S/ZETA[index])/2
        b_array[index] = -np.log(1-ZETA_S/ZETA[index])/2
        a_array[0] = 0
        b_array[0] = 0
        error = 1
        while error > 10**-6:
            for i in range(loops):
                goo, grr = back_metric(a_array, b_array)
                U_bar, epsilon = finite_differences(goo, grr, ZETA_S, ZETA, ZETA_MAX)
                error = 1
                guess_1 = -0.01
                guess_2 = -0.12
                rounds = 0
                while error > 10**-6:
                    rounds += 1
                    a_array1, b_array1 = R_K_in(U_bar, epsilon, a_array, b_array, ZETA_S, ZETA, ZETA_MAX, guess_1)
                    fx1 = a_array1[N_max-1] + b_array1[N_max-1]
                    a_array2, b_array2 = R_K_in(U_bar, epsilon, a_array, b_array, ZETA_S, ZETA, ZETA_MAX, guess_2)
                    fx2 = a_array2[N_max-1] + b_array2[N_max-1]
                    
                    guess_1, guess_2 = secant_root_alg(guess_2, fx2, guess_1, fx1)
                    print(guess_1, guess_2)
                    error = abs(a_array[N_max-1] + b_array[N_max-1])
                    print(error)
                a_array = a_array2
                b_array = b_array2
                print("Rounds: ", rounds)   
        print("Epsilon: ", epsilon)
        
        print("Final output: ", a_array[N_max-1] + b_array[N_max-1])
        plt.plot(ZETA, a_array, label=f'A of {ZETA_S}')
        plt.plot(ZETA, b_array, label =f'B of {ZETA_S}')
    plt.legend()

    
_=main()