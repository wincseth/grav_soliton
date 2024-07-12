#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:43:59 2024

@author: xaviblast123
"""

import pandas as pd
import matplotlib.pyplot as plt

#Initialize Data Frames and Arrays-----------------------------------------------------
max1, max2 = input("Which maxs would you like to compare (25, 50, 100, 120, 150)\n").split()
n1, n2 = input("And the intervals? (250, 500, 1000)\n").split()

df1 = pd.read_csv(f'datasets/n_{n1}_max_{max1}.csv')
df1_S = pd.read_csv(f'datasets/n_{n1}_max_{max1}_2.csv')

df2 = pd.read_csv(f'datasets/n_{n2}_max_{max2}.csv')
df2_S = pd.read_csv(f'datasets/n_{n2}_max_{max2}_2.csv')

ZETA = df1['ZETA'].to_numpy()
ZETA_Sn = df1_S['Unnamed: 0'].to_numpy()

# Plots to compare data----------------------------------------------------------------
plt.figure(figsize=(9,9))
plt.plot(ZETA_Sn, df1_S['Rounds'], label = f'{n1}, {max1}')
plt.plot(ZETA_Sn, df2_S['Rounds'], label = f'{n2}, {max2}')
plt.legend()
plt.savefig(f'images/{n1}_vs_{n2}_and_{max1}_vs_{max2}')