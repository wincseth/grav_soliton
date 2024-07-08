#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:17:06 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Initializing Data Frames and Arrays---------------------------------------------------
name = input("Which one are we reading in today\n")
name2 = input("And the other one\n")

df_ZETA = pd.read_csv(f'{name}.csv')
df_ZETA_S = pd.read_csv(f'{name2}.csv')

ZETA = df_ZETA['ZETA'].to_numpy()
ZETA_Sn = df_ZETA_S['Unnamed: 0'].to_numpy()
important_S = []
colors = ['red', 'orange', 'green', 'blue', 'purple']


#Main Function-------------------------------------------------------------------------
def main():
    
    print("Here are your options \n", ZETA_Sn)
    for i in range(len(colors)):
        imps = input("Give me one you need \n")
        important_S.append(imps)
    
    #Plot of U_bars
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        plt.plot(ZETA, df_ZETA[f'U Bar of {important_S[i]}'], color = colors[i], label = f'ZETA_S of {important_S[i]}')
    plt.xlim((0, 22))
    plt.xlabel('$\zeta$')
    plt.ylabel('U Bar')
    plt.title(f'U_bars_{name}')
    plt.legend()
    plt.savefig(f'U_bars_{name}.png')
    
    #Matching plots of g00 and grr
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        plt.plot(ZETA, np.exp(2*df_ZETA[f'A of {important_S[i]}']), color = colors[i], label = f'ZETA_S of {important_S[i]}')
    plt.xlim((0, 40))
    plt.xlabel('$\zeta$')
    plt.ylabel('g00')
    plt.title(f'g00_{name}')
    plt.legend()    
    plt.savefig(f'g00_{name}.png')
    
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        plt.plot(ZETA, np.exp(2*df_ZETA[f'B of {important_S[i]}']), color = colors[i], label = f'ZETA_S of {important_S[i]}')
    plt.xlim((0, 40))
    plt.xlabel('$\zeta$')
    plt.ylabel('grr')
    plt.title(f'grr_{name}')
    plt.legend()    
    plt.savefig(f'grr_{name}.png')
    
    #Epsilons and their limits
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, df_ZETA_S['Epsilons'])
    plt.xlabel('$\zeta_s$')
    plt.ylabel('Epsilon')
    plt.title(f'Epsilons_vs_ZETA_S_{name2}')
    plt.savefig(f'Epsilons_vs_ZETA_S_{name2}.png')
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, df_ZETA_S['A_0'])
    plt.xlabel('$\zeta_s$')
    plt.ylabel('A_0')
    plt.title(f'A_0_vs_ZETA_S_{name2}')
    plt.savefig(f'A_0_vs_ZETA_S_{name2}.png')
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, df_ZETA_S['E/M'])
    plt.xlabel('$\zeta_s$')
    plt.ylabel('E/M')
    plt.title(f'E_over_M_vs_ZETA_S_{name2}')
    plt.savefig(f'E_over_M_vs_ZETA_S_{name2}.png')
    
_=main()