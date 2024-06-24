#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:57:19 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
from Functions import finite_differences, back_metric
from Functions import Runge_Kutta_in_out as R_K

# Global Variables-------------------------------------------------------

# User input
n = 1000  # Interval Steps
ZETA_MAX = 50
ZETA_S = 0.1
loops = 30
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
    a_array = np.zeros_like(ZETA)
    b_array = np.zeros_like(ZETA)
    index = ZETA > ZETA_S
    a_array[index] = np.log(1-ZETA_S/ZETA[index])/2
    b_array[index] = -np.log(1-ZETA_S/ZETA[index])/2
    loops_2 = 3
    a_array[0] = 0
    b_array[0] = 0
    for j in range(loops_2):
        for i in range(loops):
            goo, grr = back_metric(a_array, b_array)
            U_bar, epsilon = finite_differences(goo, grr, ZETA_S, ZETA, ZETA_MAX)
            a_array, b_array = R_K(U_bar, epsilon, a_array, b_array, ZETA_S, ZETA, ZETA_MAX)
            print("Epsilon: ", epsilon)
        
            b_array[0] = 0
        a_array[0] = a_array[0]-(a_array[N_max-1] + b_array[N_max-1])
        print(a_array[0])

        print(a_array[N_max-1] + b_array[N_max-1])
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA, a_array, label="A")
    plt.plot(ZETA, b_array, label ='B')
    plt.legend()

    
_=main()