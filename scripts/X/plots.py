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
df_ZETA = pd.read_csv('Dependent_on_ZETA.csv')
df_ZETA_S = pd.read_csv('Dependent_on_ZETA_S.csv')

ZETA = df_ZETA['Unnamed: 0'].to_numpy()
ZETA_Sn = df_ZETA_S['Unnamed: 0'].to_numpy()
important_S = [0.01, 0.1, 0.2, 0.5, 0.74]
colors = ['red', 'orange', 'green', 'blue', 'purple']


#Main Function-------------------------------------------------------------------------
def main():
    #Plot of U_bars
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        plt.plot(ZETA, df_ZETA[f'U Bar of {important_S[i]}'], color = colors[i], label = f'ZETA_S of {important_S[i]}')
    plt.xlim((0, 22))
    plt.xlabel('$\zeta$')
    plt.ylabel('U Bar')
    plt.legend()    
    plt.savefig('U_bars.png')
    
    #Matching plots of g00 and grr
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        plt.plot(ZETA, np.exp(2*df_ZETA[f'A of {important_S[i]}']), color = colors[i], label = f'ZETA_S of {important_S[i]}')
    plt.xlim((0, 40))
    plt.xlabel('$\zeta$')
    plt.ylabel('g00')
    plt.legend()    
    plt.savefig('g00.png')
    
    plt.figure(figsize=(9,9))
    for i in range(len(important_S)):
        plt.plot(ZETA, np.exp(2*df_ZETA[f'B of {important_S[i]}']), color = colors[i], label = f'ZETA_S of {important_S[i]}')
    plt.xlim((0, 40))
    plt.xlabel('$\zeta$')
    plt.ylabel('grr')
    plt.legend()    
    plt.savefig('grr.png')
    
    #
    
    
_=main()