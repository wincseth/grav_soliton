#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:17:04 2024

@author: xaviblast123
"""

import numpy as np
from temp_functions import find_fixed_metric
from temp_functions import integrating_inside_out as integral
import pandas as pd
import sys
# Global Variables-------------------------------------------------------

# User input
n = 1500  # Interval Steps
ZETA_MAX = 20
a1 = [0.1, 0.2, 0.5, 0.7]
a2 = [0.71, 0.72, 0.73]
a3 = np.linspace(0.74, 0.7427, 5)
ZETA_Sn = np.concatenate((a1, a2, a3))
#a1 = np.linspace(0.73, 0.7427, 20)
#a1 = [0.1, 0.2, 0.5, 0.7, 0.74, 0.742, 0.7427]
#ZETA_Sn = a1
#ZETA_Sn = np.round(ZETA_Sn, decimals = 5)

ZETA_Sn = np.linspace(0.74, 0.7427, 10)
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
    zeta_s_vals = ZETA_Sn

    epsilons = []
    A_0s = []
    En_ov_M = []
    prev_g1 = -0.1
    prev_g2 = -1

    # initialize A, B values, either read from existing data or generate metric guess
    zeta_s_start = ZETA_Sn[0]
    if zeta_s_start > 0.1:
        input_name = input(f"First zeta_s value ({zeta_s_start}) is not close to zero,\n enter AB csv file name for better initial guess (w/o .csv): ")
        df0 = pd.read_csv(f"data/csv_files/{input_name}.csv")
        print("Reading file...\n")
        a_array = df0[f'A of {zeta_s_start}'].to_numpy()
        b_array = df0[f'B of {zeta_s_start}'].to_numpy()
        zeta_s_vals = np.delete(zeta_s_vals, 0)
    else:
        print("Initializing with fixed metric...\n")
        a_array, b_array = find_fixed_metric(zeta_0, zeta_s_start, ZETA)
    
    df1 = pd.DataFrame({'ZETA': ZETA})
    for i, zeta_s in enumerate(zeta_s_vals):
        U_bar, epsilon, a_array, b_array, Rounds, works = integral(a_array, b_array, ZETA, zeta_s, ZETA_MAX, prev_g1, prev_g2, zeta_0)
            
        epsilons.append(epsilon)
        A_0s.append(a_array[0])
        En_ov_M.append(1+epsilon*zeta_s*.5)
          
        df1[f'U Bar of {zeta_s}'] = abs(U_bar)
        df1[f'A of {zeta_s}'] = a_array
        df1[f'B of {zeta_s}'] = b_array
        #print(epsilons)
        #print("Final A: ", a_array[N_max-1])
        print(f'\n--- Epsilon convergence finished for zeta_s={zeta_s} ---')
        print(f"zeta_s values: {ZETA_Sn}, epsilons: {epsilons}, A0's: {A_0s}")
        
        input("\n----- Press any key to continue to next zeta_s ...\n")

    df2 = pd.DataFrame({'Epsilons' : epsilons,
                              'A_0' : A_0s,
                              'E/M' : En_ov_M}, index = zeta_s_vals,)
    
    name = input("What would you like to call these data frames? \n")
    
    df1.to_csv(f'data/csv_files/{name}.csv')
    df2.to_csv(f'data/csv_files/{name}_zeta_s.csv')
    
    


    
_=main()