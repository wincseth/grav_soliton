#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 08:59:15 2024

@author: xaviblast123
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

name = 'endpoints'
goal = 'epsilons'

df2 = pd.read_csv(f'datasets/n_1500_max_20_{name}.csv')
df2_S = pd.read_csv(f'datasets/n_1500_max_20_{name}_2.csv')

ZETA_Sn2 = df2_S['Unnamed: 0'].to_numpy()

def f(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def f2(x, a, b):
    return a*np.exp(b*x)

def f3(x, a, b, c):
    return -b*(c - x)**-a

def f4(x, a, b, c):
    return -b*(c - x)**-a
    
xdata1 = ZETA_Sn2
ydata1 = (df2_S['Epsilons'].to_numpy())
index = np.where(xdata1 > 0.739)
ydata1 = ydata1[index]
xdata1 = xdata1[index]

ydata2 = df2_S['A_0'].to_numpy()
ydata2 = ydata2[index]

initial = [1, 7, 1] 
bounds = (0, [20, 20, 5])

popt, pcov = curve_fit(f3, xdata1, ydata1, p0 = initial, maxfev=15000, bounds = bounds)

popt2, pcov2 = curve_fit(f4, xdata1, ydata2, maxfev=15000)

a, b, c = popt
print(f'For Epsilons, Coefficients: a = {a}, b={b}, c = {c}')

residuals = ydata1- f3(xdata1, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata1-np.mean(ydata1))**2)
r_squared = 1 - (ss_res / ss_tot)

print("R^2 value is: ", r_squared)

a2, b2, c2 = popt2
print(f'For A[0], Coefficients: a = {a2}, b={b2}, c = {c2}')

residuals = ydata2- f4(xdata1, *popt2)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((ydata1-np.mean(ydata1))**2)
r_squared = 1 - (ss_res / ss_tot)

print("R^2 value is: ", r_squared)

plt.figure(figsize=(9,9))
plt.scatter(xdata1, ydata1, label=f'{name} Data')
plt.plot(xdata1, f3(xdata1, *popt), label='Fitted line', color='red')
plt.xlabel('$\zeta_s$')
plt.ylabel('Epsilon')
plt.title(f'{name} Epsilon vs ZETA_S')

plt.legend()
#plt.savefig(f'images/{name}_fit_{goal}.png')

plt.figure(figsize=(9,9))
plt.scatter(xdata1, ydata2, label=f'{name} Data')
plt.plot(xdata1, f4(xdata1, *popt2), label='Fitted line', color='red')
plt.xlabel('$\zeta_s$')
plt.ylabel('A[0]')
plt.title(f'{name} A[0] vs ZETA_S')

plt.legend()
plt.savefig(f'images/{name}_fit_A[0].png')