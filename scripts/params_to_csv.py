#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from kg_and_metric_functions_g00_grr import find_fixed_metric
from kg_and_metric_functions_g00_grr import iterate_kg_and_metric
import pandas as pd

# Global Variables-------------------------------------------------------

# User inputh
N = 1500  # Interval Steps
ZETA_MAX = 20
#ZETA_Sn = [0.1, 0.2, 0.5, 0.7, 0.74, 0.7427]

a1 = np.linspace(0.1, 0.7, 7)
a2 = [0.71, 0.72, 0.73]
a3 = np.linspace(0.74, 0.7427, 6)
ZETA_Sn = np.concatenate((a1, a2, a3))
ZETA_Sn = np.round(ZETA_Sn, decimals = 5)


# -----------------------------------------------------------------------
# Global variables afterwards
ZETA_0 = 20
DEL = (ZETA_MAX)/(N + 1)  # Spacing of intervals
ZETA = np.arange(0, ZETA_MAX, DEL)  # ZETA array
N_MAX = len(ZETA)
G = 6.7*10**(-39)  # normalized gravity
M_PL = 1 / np.sqrt(G)  # mass of plank mass
M = 8.2*10**10  # if a equals the atomic Bohr radius
a = 1 / (G*M**3)  # gravitational bohr radius
R_S = 2*G*M  # schwarzschild radius

#Main Function---------------------------------------------------------
def main():

    # define output arrays
    epsilons = []
    A_0s = []
    En_ov_M = []
    
    # initialize A, B values, either read from existing data or generate metric guess
    zeta_s_start = ZETA_Sn[0]
    if zeta_s_start > 0.1:
        input_name = input(f"First zeta_s value ({zeta_s_start}) is not close to zero,\n enter AB csv file name for better initial guess (w/o .csv): ")
        df0 = pd.read_csv(f"data/csv_files/{input_name}.csv")
        print("Reading file...\n")
        a_array = df0[f'A of {zeta_s_start}'].to_numpy()
        b_array = df0[f'B of {zeta_s_start}'].to_numpy()
    else:
        print("Initializing with fixed metric...\n")
        a_array, b_array = find_fixed_metric(ZETA_0, zeta_s_start, ZETA)

    # loop through zeta_s values
    for i, zeta_s in enumerate(ZETA_Sn):
        
        # iterate kg/metric for converging epsilon
        A_0_guess = a_array[0]
        U_bar, epsilon, a_array, b_array, R_tilde, eps_rounds, working_conv = iterate_kg_and_metric(a_array, b_array, ZETA, zeta_s, ZETA_MAX, A_0_guess, ZETA_0)
            
        epsilons.append(epsilon)
        A_0s.append(a_array[0])
        En_ov_M.append(1+epsilon*zeta_s*.5)
        
        print(f'\n--- Epsilon convergence finished for zeta_s={zeta_s} ---')
        print(f"zeta_s values: {ZETA_Sn}, epsilons: {epsilons}, A0's: {A_0s}")

        if i == 0:
            df1 = pd.DataFrame({'ZETA': ZETA})
        df1[f'U Bar of {zeta_s}'] = abs(U_bar)
        df1[f'A of {zeta_s}'] = a_array
        df1[f'B of {zeta_s}'] = b_array
        df1[f'R tilde of {zeta_s}'] = R_tilde
        #print(epsilons)
        #print("Final A: ", a_array[N_MAX-1])

    df2 = pd.DataFrame({'Epsilons' : epsilons,
                              'A_0' : A_0s,
                              'E/M' : En_ov_M}, index = ZETA_Sn,)
    print("--- Dataset generation conplete ---\n")
    name = input("Enter name for CSV holding u bar, A, B, R tilde (no .csv): \n")
    name2 = input("Enter name for CSV holding epsilon, A0, E/M (no .csv): \n")
    
    df1.to_csv(f"data/{name}.csv")
    df2.to_csv(f"data/{name2}.csv")
    
    
_=main()