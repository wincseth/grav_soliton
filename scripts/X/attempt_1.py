#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:36:13 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt

A = 0
B = 0
zetta_s = 0.01
n = 20
delta = 1/(n+1)
zetta = n*delta
x = np.linspace(-5, 5, 2)  # 100 points from -5 to 5
y = np.linspace(-5, 5, 2)  # 100 points from -5 to 5
X, Y = np.meshgrid(x, y)


#Finite differences attempt 1
def finite_differences(A, B):
    goo = np.exp(2*A) #g_00 guess
    grr = np.exp(2*B) #g_rr guess
    goo_approx = 2*np.exp(A)*np.sinh(A) #approx. used for g_00(1-1/g_00)
    C = -(goo/grr)*(zetta_s**2/((4*zetta**2)*(zetta**2-zetta_s**2)))+(2/zetta_s)*goo_approx+2*(goo/grr)/delta**2 #first value
    D = -(goo/grr)/delta**2 #second value
    matrix = [[C, D],[D, C]]
    lambdan = np.linalg.eig(matrix)
    epsilon = lambdan[0]/(1+np.sqrt(1+zetta_s*lambdan[0]/2))
    return [epsilon, lambdan[1]]

#now, attempting RK
def Runge_Kutta():
    A_0 = B_0 = 0
    A = A_0
    B = B_0
    R_tilde = 1
    for i in range(n):
        goo = np.exp(2*A)
        grr = np.exp(2*B)
        A_temp = A + (1/(2*zetta))*(grr-1)-(zetta_s*i*delta/4)*grr*R_tilde**2+(zetta_s**2*(i*delta)/8)

plt.figure(figsize=(9,9))
plt.contourf(X, Y, finite_differences(A, B)[1], 20)