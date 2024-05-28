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
n = 1
DELTA = 1/(n+1)
zetta = n*DELTA


#Finite differences attempt 1
def finite_differences(A, B):
    goo = np.exp(2*A) #g_00 guess
    grr = np.exp(2*B) #g_rr guess
    goo_approx = 2*np.exp(A)*np.sinh(A) #approx. used for g_00(1-1/g_00)
    C = -(goo/grr)*(zetta_s**2/((4*zetta**2)*(zetta**2-zetta_s**2)))+(2/zetta_s)*goo_approx+2*(goo/grr)/DELTA**2 #first value
    D = -(goo/grr)/DELTA**2 #second value
    matrix = [[C, D],[D, C]]
    lambdan = np.diag(matrix)
    epsilon = lambdan/(1+np.sqrt(1+zetta_s*lambdan/2))
    return epsilon 

#now, attempting RK
def Runge_Kutta():
    A_0 = B_0 = 0
    A = A_0
    B = B_0
    goo = np.exp(2*A)
    grr = np.exp(2*B)
    A_temp = A_0 + (1/(2*zetta))*(grr-1)-(zetta_s*zetta/4)*grr

print(finite_differences(A, B))