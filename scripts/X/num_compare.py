#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 19:46:33 2024

@author: xaviblast123
"""

import pandas as pd
import numpy as np

df1 = pd.read_csv('datasets/n_1500_max_20_finalend.csv')
df2 = pd.read_csv('datasets/n_1500_max_20_seth7.csv')
df3 = pd.read_csv('datasets/n_1500_max_20_finalend_2.csv')
df4 = pd.read_csv('datasets/n_1500_max_20_seth7_2.csv')

ZETA_S1 = df3['Unnamed: 0'].to_numpy()
ZETA_S2 = df4['Unnamed: 0'].to_numpy()

print("XAVI CODE: \n")
for i in range(len(ZETA_S1)):
    temp = df1[f'A of {ZETA_S1[i]}'].to_numpy()
    temp2 = df1[f'B of {ZETA_S1[i]}'].to_numpy()
    print(f'For XAVI CODE at {ZETA_S1[i]}, {temp[1500]} + {temp2[1500]} ', temp[1500] + temp2[1500])
    temp3 = df2[f'A of {ZETA_S2[i]}'].to_numpy()
    temp4 = df2[f'B of {ZETA_S2[i]}'].to_numpy()
    print(f'For SETH CODE at {ZETA_S1[i]}, {temp3[1500]} + {temp4[1500]} ', temp3[1500] + temp4[1500])
    