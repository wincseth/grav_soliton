#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:26:42 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Functions import finding_average as f_a

#Initializing Data Frames and Arrays---------------------------------------------------
name = input("Which one are we reading in today\n")

df_ZETA = pd.read_csv(f'datasets/{name}.csv')
df_ZETA_S = pd.read_csv(f'datasets/{name}_2.csv')

ZETA = df_ZETA['ZETA'].to_numpy()
ZETA_Sn = df_ZETA_S['Unnamed: 0'].to_numpy()
important_S = [0.1, 0.5, 0.7, 0.7427]
colors = ['red', 'orange', 'green', 'blue', 'cyan', 'purple',  'black']

#Code for Plots-----------------------------------------------------------------------
def main():
    plt.figure(figsize=(9, 9))
    for i in range(len(important_S)):
        u_bar_col = f'U Bar of {important_S[i]}'
        y = abs(df_ZETA[u_bar_col].to_numpy())
        x = f_a(ZETA, y)
        x = np.round(x, 4)
        idx = (np.abs(ZETA - x)).argmin()
        z = y[idx]
        plt.scatter(x, z, color=colors[i], zorder = 5)
        plt.plot(ZETA, y, color=colors[i], label=f'$\zeta_S$ = {important_S[i]}')
        plt.annotate(f'for $\zeta_S$ = {important_S[i]}, <$\zeta$> is {x}', xy=(x, z), textcoords="offset points", xytext=(3, 5))
        plt.xlabel('$\zeta$')
        plt.title('<$\zeta$> for Probability Distributions')
        plt.ylabel('U bar')
    plt.legend()
    plt.savefig("images/expected_zeta.png")

main()
