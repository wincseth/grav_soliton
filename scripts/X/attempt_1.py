#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:36:13 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt

A = 1.3
B = -.2
zetta_s = 0.01
n = 8
delta = 1/(n+1)
zetta = n*delta
x = np.linspace(-10, 10, n)  
y = np.linspace(-10, 10, n)
X, Y = np.meshgrid(x, y)


#Finite differences attempt 1
def finite_differences(A, B):
    goo = np.exp(2*A) #g_00 guess
    grr = np.exp(2*B) #g_rr guess
    goo_approx = 2*np.exp(A)*np.sinh(A) #approx. used for g_00(1-1/g_00)
    C = -(goo/grr)*(zetta_s**2/((4*zetta**2)*(zetta**2-zetta_s**2)))+(2/zetta_s)*goo_approx+2*(goo/grr)/delta**2 #first value
    D = -(goo/grr)/delta**2 #second value
    matrix = np.zeros((n, n)) #creating a matrix of 0s
    for i in range(n): #adding the values into 0 for all the equations from 0 to n
        matrix[i, i] = C
        if i==0:
            matrix[i, i+1] = D
        else:
            matrix[i, i-1] = D
    eigenvalues, eigenvectors = np.linalg.eig(matrix) #getting the eigenvalues and eigenvectors
    eig_vec = eigenvectors[:, 0].reshape((n, 1))
    eig_vec = np.tile(eig_vec, (1, n)) / np.sqrt(np.abs(X) + 1e-10) #reshaping and normalizing
    epsilon = eigenvalues/(1+np.sqrt(1+zetta*eigenvalues/2))
    return [epsilon, eig_vec]

#now, attempting RK
#def Runge_Kutta():
    #A_0 = B_0 = 0
    #A = A_0
    #B = B_0
    #for i in range(n):
        #goo = np.exp(2*A)
        #grr = np.exp(2*B)
        #A_temp = A + (1/(2*zetta))*(grr-1)-(zetta_s*i*delta/4)*grr*R_tilde**2+(zetta_s**2*(i*delta)/8)

print(finite_differences(A, B)[0])
plt.figure(figsize=(9,9))
plt.contourf(X, Y, finite_differences(A, B)[1]**2, 20) #plotting to visualize probability density