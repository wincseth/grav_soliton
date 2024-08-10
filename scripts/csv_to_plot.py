#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:17:06 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as r

#Initializing Data Frames and Arrays---------------------------------------------------
name = input("Enter csv file name for data w.r.t. zeta: ")
name2 = input("Enter csv file name for data w.r.t. zeta_s: ")

df_ZETA = pd.read_csv(f'data/n_2500_20_steps_to_0.7427_tol=1e-7/{name}.csv')
df_ZETA_S = pd.read_csv(f'data/n_2500_20_steps_to_0.7427_tol=1e-7/{name2}.csv')

ZETA = df_ZETA['ZETA'].to_numpy()
ZETA_Sn = df_ZETA_S['Unnamed: 0'].to_numpy()
#colors = ['red', 'orange', 'green', 'cyan', 'blue', 'purple',  'black']

#Main Function-------------------------------------------------------------------------
def main():
    important_S = []
    colors = [(r.random(), r.random(), r.random()) for _ in range(len(df_ZETA_S))]
    
    print("zeta_s values in file:\n", ZETA_Sn)
    choose_zeta_s = input("Plot all zeta_s values? (Y/N): ")
    if choose_zeta_s == 'N':
        num_colors = input("Enter number of zeta_s values to plot:")
        colors = [(r.random(), r.random(), r.random()) for _ in range(int(num_colors))]
        for i, color in enumerate(colors):
            imps = input(f"(zeta_s number {i}) Enter an above zeta_s to plot with: ")
            important_S.append(imps)
    elif choose_zeta_s == 'Y':
        important_S = ZETA_Sn
    
    #Plot of U_bars
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        plt.plot(ZETA, df_ZETA[f'U Bar of {important_S[i]}'], color = colors[i], label = f'ZETA_S of {important_S[i]}', alpha=0.5, marker='.')
    plt.xlim((0, 22))
    plt.xlabel('$\zeta$')
    plt.ylabel('U Bar')
    plt.title(f'U_bars from {name}.csv')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'images/U_bars_{name}.png')
    
    #Matching plots of g00 and grr
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        #plt.plot(ZETA, np.exp(2*df_ZETA[f'A of {important_S[i]}']), color = colors[i], label = f'ZETA_S of {important_S[i]}')
        plt.plot(ZETA, df_ZETA[f'A of {important_S[i]}'], color = colors[i], label = f'ZETA_S={important_S[i]}', alpha=0.5, marker='.')
    plt.xlim((0, 40))
    plt.xlabel('$\zeta$')
    plt.ylabel('A')
    plt.title(f'A from {name}.csv')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'images/g00_{name}.png')
    
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        #plt.plot(ZETA, np.exp(2*df_ZETA[f'B of {important_S[i]}']), color = colors[i], label = f'ZETA_S of {important_S[i]}')
        plt.plot(ZETA, df_ZETA[f'B of {important_S[i]}'], color = colors[i], label = f'ZETA_S={important_S[i]}', alpha=0.5, marker='.')
    plt.xlim((0, 40))
    plt.xlabel('$\zeta$')
    plt.ylabel('B')
    plt.title(f'B from {name}.csv')
    plt.legend()
    plt.grid(True)    
    #plt.savefig(f'images/grr_{name}.png')
    
    #Epsilons and their limits
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, df_ZETA_S['Epsilons'], alpha=0.5, marker='.')
    plt.xlabel('$\zeta_s$')
    plt.ylabel('Epsilon')
    plt.title(f'Epsilons_vs_ZETA_S from {name2}.csv')
    plt.grid(True)
    plt.xlim(0.6,0.8)
    #plt.savefig(f'images/Epsilons_vs_ZETA_S_{name2}.png')
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, df_ZETA_S['A_0'], alpha=0.5, marker='.')
    plt.xlabel('$\zeta_s$')
    plt.ylabel('A_0')
    plt.title(f'A_0_vs_ZETA_S from {name2}.csv')
    plt.grid(True)
    #plt.savefig(f'images/A_0_vs_ZETA_S_{name2}.png')
    
    plt.figure(figsize=(9,9))
    plt.plot(ZETA_Sn, df_ZETA_S['E/M'], alpha=0.5, marker='.')
    plt.xlabel('$\zeta_s$')
    plt.ylabel('E/M')
    plt.title(f'E_over_M_vs_ZETA_S from {name2}.csv')
    plt.grid(True)
    #plt.savefig(f'images/E_over_M_vs_ZETA_S_{name2}.png')
    
    plt.show()
_=main()