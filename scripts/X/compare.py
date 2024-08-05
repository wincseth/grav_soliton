#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:43:59 2024

@author: xaviblast123
"""

import pandas as pd
import matplotlib.pyplot as plt

#Initialize Data Frames and Arrays-----------------------------------------------------


df1 = pd.read_csv('datasets/n_1500_max_20_seth8.csv')
df1_S = pd.read_csv('datasets/n_1500_max_20_seth8_2.csv')

df2 = pd.read_csv('datasets/n_1500_max_20_finalend.csv')
df2_S = pd.read_csv('datasets/n_1500_max_20_finalend_2.csv')

ZETA_Sn1 = df1_S['Unnamed: 0'].to_numpy()
ZETA_Sn2 = df2_S['Unnamed: 0'].to_numpy()

# Plots to compare data----------------------------------------------------------------
plt.figure(figsize=(9,9))
plt.plot(ZETA_Sn1, df1_S['Epsilons'], label = 'Seth', marker = 'o')
plt.plot(ZETA_Sn2, df2_S['Epsilons'], label = 'Xavi', marker = 'o')
plt.xlim((.74, .7433))
plt.ylim((-0.325, -0.352))
plt.legend()