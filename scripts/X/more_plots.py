#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 19:55:07 2024

@author: xaviblast123
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    n_types = [1000, 1500, 2000, 2500]
    final_ZETAS = [0.742431, 0.742712, 0.742761]
    df1 = pd.DataFrame()
    
    for i in range(len(n_types)):
        data = pd.read_csv(f'datasets/n_{n_types[i]}_max_20_imp_2.csv', usecols=['Epsilons'])
        if i == 0:
            ZETA_S = pd.read_csv(f'datasets/n_{n_types[i]}_max_20_imp_2.csv', usecols=['Unnamed: 0'])
            df1['ZETA_S'] = ZETA_S.squeeze()
        df1[f'{n_types[i]}'] = data['Epsilons']
    
    for j in range(len(ZETA_S)):
        plt.figure(figsize=(9,9))
        y = df1.iloc[j].to_numpy()
        print(y)
        y_title = y[0]
        y = np.delete(y, 0)
        plt.plot(n_types, y)
        plt.title(f'ZETA_S of {y_title}')
        plt.xlabel('Iterations')
        plt.ylabel('Epsilon')
        plt.savefig(f'images/conv_of_epsilon_{y_title}.png')

main()
