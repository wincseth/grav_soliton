#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:57:19 2024

@author: xaviblast123
"""

import numpy as np
import pandas as pd
from Functions import integrating_inside_out as integral
# Global Variables-------------------------------------------------------

# User input
n = 500  # Interval Steps
ZETA_MAX = 20
ZETA_Sn = [0.7423, 0.7433]
df1 = pd.read_csv(f'n_{n}_max_{ZETA_MAX}.csv')
# -----------------------------------------------------------------------
# Global variables afterwards
zeta_0 = 10
DEL = (ZETA_MAX)/(n + 1)  # Spacing of intervals
ZETA = np.arange(0, ZETA_MAX, DEL)  # ZETA array
N_max = len(ZETA)
G = 6.7*10**(-39)  # normalized gravity
M_PL = 1 / np.sqrt(G)  # mass of plank mass
M = 8.2*10**10  # if a equals the atomic Bohr radius
a = 1 / (G*M**3)  # gravitational bohr radius
R_S = 2*G*M  # schwarzschild radius

#Main Function---------------------------------------------------------
#ZETA_MAX = 20, n = 500 ZETA_S = 0.742431, n = 1000 ZETA_S = 0.742712, n = 1500 ZETA_S = 0.742761
def main():
    works = True
    prev_g1 = -0.1
    prev_g2 = -1
    diff = ZETA_Sn[1] - ZETA_Sn[0]
    ZETA_SOUT = ZETA_Sn[1]
    ZETA_SIN = ZETA_Sn[0]
    i = 0
    while diff > 10**-6:
        print('i is ', i)
        ZETA_S = ZETA_Sn[i]
        if i == 0:
            a_array = df1['A of 0.7'].to_numpy()
            b_array = df1['B of 0.7'].to_numpy()
            
        prev_a_array = a_array
        prev_b_array = b_array
        U_bar, epsilon, a_array, b_array, rounds, works = integral(a_array, b_array, ZETA, ZETA_S, ZETA_MAX, prev_g1, prev_g2, zeta_0)
        
        if works == False:
            ZETA_SOUT = ZETA_S
            print(i, U_bar, epsilon, a_array, b_array, rounds)
            a_array = prev_a_array
            b_array = prev_b_array
            ZETA_Sn.append((ZETA_SOUT+ZETA_SIN)/2)
            
        if works == True and i > 0:
            ZETA_SIN = ZETA_S
            ZETA_Sn.append((ZETA_SOUT+ZETA_SIN)/2)
            
        print(epsilon)
        print("Final A: ", a_array[N_max-1])
        print(ZETA_Sn)
        diff = ZETA_SOUT - ZETA_SIN
        i += 1

    
_=main()