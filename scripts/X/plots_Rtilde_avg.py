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
from scipy.optimize import curve_fit

#Initializing Data Frames and Arrays---------------------------------------------------
name = 'n_1500_max_20_endpoints'

df_ZETA = pd.read_csv(f'datasets/{name}.csv')
df_ZETA_S = pd.read_csv(f'datasets/{name}_2.csv')

ZETA = df_ZETA['ZETA'].to_numpy()
ZETA_Sn = df_ZETA_S['Unnamed: 0'].to_numpy()
index = np.where(ZETA_Sn > 0.739)
ZETA_Sn = ZETA_Sn[index]
important_S = ZETA_Sn
colors = ['red', 'orange', 'green', 'blue', 'cyan', 'purple',  'black']

#Code for Plots-----------------------------------------------------------------------
def main():
    plt.figure(figsize=(9, 9))
    avg = []
    for i in range(len(important_S)):
        u_bar_col = f'U Bar of {important_S[i]}'
        y = abs(df_ZETA[u_bar_col].to_numpy())
        x = f_a(ZETA, y)
        avg.append(x)
        x = np.round(x, 4)
        #idx = (np.abs(ZETA - x)).argmin()
        #z = y[idx]
        #plt.scatter(x, z, color=colors[i], zorder = 5)
        #plt.plot(ZETA, y, color=colors[i], label=f'$\zeta_S$ = {important_S[i]}')
        #plt.annotate(f'for $\zeta_S$ = {important_S[i]}, <$\zeta$> is {x}', xy=(x, z), textcoords="offset points", xytext=(3, 5))
        #plt.xlabel('$\zeta$')
        #plt.title('<$\zeta$> for Probability Distributions')
        #plt.ylabel('U bar')
    #plt.legend()
    
    def f4(x, a, b, c):
        return b*(c - x)**a
    
    initial = [1, 7, 1] 
    bounds = (0, [20, 30, 1])
    
    popt, pcov = curve_fit(f4, ZETA_Sn, avg, p0 = initial, maxfev=15000, bounds = bounds)
    
    a, b, c = popt
    print(f'For <ZETA>, Coefficients: a = {a}, b={b}, c = {c}')
    
    residuals = avg- f4(ZETA_Sn, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((avg-np.mean(avg))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print("R^2 value is: ", r_squared)
    
    plt.figure(figsize=(9,9))
    plt.scatter(ZETA_Sn, avg)
    plt.plot(ZETA_Sn, f4(ZETA_Sn, *popt), label='Fitted line', color='red')
    

main()
