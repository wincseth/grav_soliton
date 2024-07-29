#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:37:33 2024

@author: xaviblast123
"""

import numpy as np
from Functions import find_fixed_metric
from Functions import integrating_inside_out as integral
# Global Variables-------------------------------------------------------

# User input
n = 1000  # Interval Steps
ZETA_MAX = 20
ZETA_Sn = [0.74, 0.743]
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
    prev_g1 = -0.1
    prev_g2 = -1
    ZETA_START = ZETA_Sn[0]
    ZETA_END = ZETA_Sn[1]
    a_array, b_array, h_tilde = find_fixed_metric(zeta_0, ZETA_Sn[0], ZETA)
    tol = 1
    i = 0
    while tol > 10**-6:
        U_bar, epsilon, a_array, b_array, Rounds, works, U_tilde = integral(a_array, b_array, ZETA, ZETA_Sn[i], ZETA_MAX, prev_g1, prev_g2, zeta_0)
        if works == True and i != 0:
            ZETA_START = ZETA_Sn[i]
            ZETA_Sn.append((ZETA_END+ZETA_START)/2)
            print(ZETA_Sn)
        if works == False:
            ZETA_END = ZETA_Sn[i]
            ZETA_Sn.append((ZETA_END+ZETA_START)/2)
            print(ZETA_Sn)
        tol = ZETA_END - ZETA_START
        i += 1
    print(ZETA_Sn)
    


    
_=main()