#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:17:04 2024

@author: xaviblast123
"""

import numpy as np
from Functions import find_fixed_metric
from Functions import integrating_inside_out as integral
import pandas as pd
# Global Variables-------------------------------------------------------

# User input
n = 2000  # Interval Steps
ZETA_MAX = 20
ZETA_Sn = [0.1, 0.2, 0.5, 0.7, 0.74, 0.742, 0.7427]
ZETA_Sn = np.round(ZETA_Sn, decimals = 5)
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
    epsilons = []
    A_0s = []
    En_ov_M = []
    prev_g1 = -0.1
    prev_g2 = -1
    for i in range(len(ZETA_Sn)):
        ZETA_S = ZETA_Sn[i]
        if i == 0:
            a_array, b_array, h_tilde = find_fixed_metric(zeta_0, ZETA_S, ZETA)
        U_bar, epsilon, a_array, b_array, Rounds, works = integral(a_array, b_array, ZETA, ZETA_S, ZETA_MAX, prev_g1, prev_g2, zeta_0)
            
        epsilons.append(epsilon)
        A_0s.append(a_array[0])
        En_ov_M.append(1+epsilon*ZETA_S*.5)
        
        if i == 0:
            df1 = pd.DataFrame({'ZETA': ZETA})
        df1[f'U Bar of {ZETA_S}'] = abs(U_bar)
        df1[f'A of {ZETA_S}'] = a_array
        df1[f'B of {ZETA_S}'] = b_array
        print(epsilons)
        print("Final A: ", a_array[N_max-1])

    df2 = pd.DataFrame({'Epsilons' : epsilons,
                              'A_0' : A_0s,
                              'E/M' : En_ov_M}, index = ZETA_Sn,)
    
    name = input("What would you like to call these data frames? \n")
    name2 = input("And the second one?\n")
    
    df1.to_csv(name)
    df2.to_csv(name2)
    
    


    
_=main()