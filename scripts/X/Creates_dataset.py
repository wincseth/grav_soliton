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
n = 1500  # Interval Steps
ZETA_MAX = 20
a1 = np.linspace(0.1, 0.7, 7)
a2 = [0.71, 0.72, 0.73]
a3 = np.linspace(0.74, 0.7427, 20)_max_20
ZETA_Sn = np.concatenate((a1, a2, a3))
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
    ZETA_S = ZETA_Sn[0]
    a_array, b_array, h_tilde = find_fixed_metric(zeta_0, ZETA_S, ZETA)
    df1 = pd.DataFrame({'ZETA': ZETA})
    for i in range(len(ZETA_Sn)):
        ZETA_S = ZETA_Sn[i]
        U_bar, epsilon, a_array, b_array, Rounds, works = integral(a_array, b_array, ZETA, ZETA_S, ZETA_MAX, prev_g1, prev_g2, zeta_0)
            
        epsilons.append(epsilon)
        A_0s.append(a_array[0])
        En_ov_M.append(1+epsilon*ZETA_S*.5)
          
        df1[f'U Bar of {ZETA_S}'] = abs(U_bar)
        df1[f'A of {ZETA_S}'] = a_array
        df1[f'B of {ZETA_S}'] = b_array
        print(epsilons)
        print("Final A: ", a_array[N_max-1])

    df2 = pd.DataFrame({'Epsilons' : epsilons,
                              'A_0' : A_0s,
                              'E/M' : En_ov_M}, index = ZETA_Sn,)
    
    name = input("What would you like to call these data frames? \n")
    
    df1.to_csv(f'datasets/{name}.csv')
    df2.to_csv(f'datasets/{name}_2.csv')
    
    


    
_=main()