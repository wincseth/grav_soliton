#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:36:13 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt

A = .8
B = -.5
zetta_s = 0.01
n = 50
delta = 1/(n+1)
zetta = n*delta
x = np.linspace(-100, 100, n)  
y = np.linspace(-100, 100, n)
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
        if i < n-1:
            matrix[i, i+1] = D
        if i > 1:
            matrix[i, i-1] = D
    eigenvalues, eigenvectors = np.linalg.eig(matrix) #getting the eigenvalues and eigenvectors
    eig_vec = eigenvectors[:, 1].reshape((n, 1))
    eig_vec = np.tile(eig_vec, (1, n)) / np.sqrt(np.abs(X) + 1e-10) #reshaping and normalizing
    eigen_vec = eigenvectors[:, 1]
    epsilon = eigenvalues/(1+np.sqrt(1+zetta*eigenvalues/2))
    return [epsilon, eig_vec, eigen_vec]

def radial_function(u):
    R = np.sqrt(np.exp(2*A)/np.exp(2*B))*u**2 / zetta**2
    return R

def der_of_R(R):
    dR = np.zeros(len(R))
    for i in range(len(R)):
        if i == n-1 or i == 0:
            dR[i]=0
            continue
        dR[i] = (R[i+1]-R[i-1])/(2*delta)
    return(dR)

#now, attempting RK
#def Runge_Kutta():
    #A_0 = B_0 = 0
    #A = A_0
    #B = B_0
    #for i in range(n):
        #goo = np.exp(2*A)
        #grr = np.exp(2*B)
        #A_temp = A + (1/(2*zetta))*(grr-1)-(zetta_s*i*delta/4)*grr*R_tilde**2+(zetta_s**2*(i*delta)/8)

print(finite_differences(A, B))
R = radial_function(finite_differences(A, B)[2])
print(R)
dR = der_of_R(R)
print(dR)
plt.figure(figsize=(9,9))
plt.contourf(X, Y, finite_differences(A, B)[1]**2, n) #plotting to visualize probability density
plt.savefig('Probability density')
plt.figure(figsize=(9,7))
plt.plot(x, R)
plt.savefig('R Function')
plt.figure(figsize=(9,7))
plt.plot(x, dR)
plt.savefig('Derivative of R')