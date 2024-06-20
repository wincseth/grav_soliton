#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:18:35 2024

@author: xaviblast123
"""

#Imported Libraries-----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from Functions import finite_differences as F_D
from Functions import Runge_Kutta as R_K

#Global Variables-------------------------------------------

# User input
n = 500  # Interval Steps
ZETA_MAX = 50
ZETA_Sn = [0.4]
ZETA_0 = 0.5
loops = 1
# ----------------------------------------------------------
# Global variables afterwards
DEL = (ZETA_MAX)/(n + 1)  # Spacing of intervals
ZETA = np.arange(0, ZETA_MAX, DEL)  # ZETA array
N_max = len(ZETA)
G = 6.7*10**(-39)  # normalized gravity

M_PL = 1 / np.sqrt(G)  # mass of plank mass
M = 8.2*10**10  # if a equals the atomic Bohr radius
a = 1 / (G*M**3)  # gravitational bohr radius
R_S = 2*G*M  # schwarzschild radius

#Functions---------------------------------------------------

def metric_check(ZETA, ZETA_S):
    '''
    New metric to test out
    
    Parameters:
        ZETA: 1d np array
    Outputs:
        goo: 1D np array
        grr: 1D np array
    '''
    goo = np.ones_like(ZETA)
    grr = np.ones_like(ZETA)
    
    index = np.where(ZETA < ZETA_0)
    f = (1-(ZETA_S/(ZETA_0**3))*ZETA**2)
    f_0 = 1-(ZETA_S/ZETA_0)
    goo[index] = ((3*np.sqrt(f_0) - np.sqrt(f[index]))**2)/4
    grr[index] = 1/f[index]
    
    index2 = np.where(ZETA >= ZETA_0)
    f2 = 1 - ZETA_S/ZETA[index2]
    goo[index2] = f2
    grr[index2] = 1/f2
    
    return goo, grr
    
#Main Function----------------------------------------------

def main():
    plt.figure(figsize=(9,9))
    for i in range(len(ZETA_Sn)):
        goo, grr = metric_check(ZETA, ZETA_Sn[i])
        for j in range(loops):
            U_bar, epsilon, e_val = F_D(goo, grr, ZETA_Sn[i])
            print(epsilon)
            A_array, B_array = R_K(U_bar, epsilon, goo, grr, ZETA_Sn[i])
            goo = np.exp(2*A_array)
            grr = np.exp(2*B_array)
    
        plt.plot(ZETA, A_array, label=f'A of {ZETA_Sn[i]}', color = (1, i/(len(ZETA_Sn)), i/len(2*ZETA_Sn)))
        plt.plot(ZETA, B_array, label=f'B of {ZETA_Sn[i]}', color = (1, i/(len(ZETA_Sn)), i/len(2*ZETA_Sn)))
    plt.legend()
    ax1 = plt.gca()
    ax1.set_xlim([0, 20])

_=main()