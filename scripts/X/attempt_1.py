#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:36:13 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt

n = 500
A = np.linspace(0, 3, n)
B = np.linspace(0, 2, n) 
zetta_s = .01
delta = 1/(n+1)
zetta_max = n*delta
zetta_n = np.arange(1*10**-10, zetta_max, delta)
Rounds = 20

#Finite differences attempt 1
def finite_differences(A, B):
    goo = np.exp(2*A) #g_00 guess
    grr = np.exp(2*B) #g_rr guess
    goo_approx = 2*np.exp(A)*np.sinh(A) #approx. used for g_00(1-1/g_00)
    matrix = np.zeros((n, n)) #creating a matrix of 0s
    for i in range(n): #adding the values into 0 for all the equations from 0 to n
        C = -(goo[i]/grr[i])*(zetta_s**2/((4*zetta_n[i]**2)*(zetta_n[i]**2-zetta_s**2)))+(2/zetta_s)*goo_approx[i]+2*(goo[i]/grr[i])/delta**2 #first value
        matrix[i, i] = C
        if i < n-1:
            D = -np.sqrt(goo[i]/grr[i])*np.sqrt(goo[i+1]/grr[i+1])/(delta)**2 #second value
            matrix[i, i+1] = D
        if i > 0:
            E = -np.sqrt(goo[i]/grr[i])*np.sqrt(goo[i-1]/grr[i-1])/(delta)**2
            matrix[i, i-1] = E
    eigenvalues, eigenvectors = np.linalg.eig(matrix) #getting the eigenvalues and eigenvectors
    N = np.argmin(eigenvalues)
    print(N, eigenvalues[N])
    epsilon = eigenvalues[N]/(1+np.sqrt(1+zetta_s*eigenvalues[N]/2))
    eigen_vec = eigenvectors[:, N]
    return [epsilon, eigen_vec]

def radial_function(u, A):
    R = np.sqrt(np.exp(2 * A)) * u / (n * delta)
    R = np.sqrt(R**2)/(np.sqrt(np.sum(R**2 * delta)))
    return R

def der_of_R(R):
    dR = np.zeros(len(R))
    for i in range(len(R)):
        if i == n-1 or i == 0:
            dR[i]=0
            continue
        dR[i] = (R[i+1]-R[i-1])/(zetta_n[i+1]-zetta_n[i-1])
    return(dR)

def temp(A, B, zetta, R, dR, epsilon):
    goo = np.exp(2*A)
    grr = np.exp(2*B)
    A_temp =(1/(2*zetta))*(grr-1)-(zetta_s*zetta/4)*grr*R**2+(zetta_s**2*zetta/8)*dR**2+(zetta_s*zetta/4)*(1+zetta_s*epsilon/2)**2*(grr/goo)*R**2
    B_temp =-(1/(2*zetta))*(grr-1)+(zetta_s*zetta/4)*grr*R**2+(zetta_s**2*zetta/8)*dR**2+(zetta_s*zetta/4)*(1+zetta_s*epsilon/2)**2*(grr/goo)*R**2
    return [A_temp, B_temp]

#now, attempting RK
def Runge_Kutta(R, dR, A, B, epsilon):
    for i in range(len(R)-1):
        A_temp = A[i] + delta*temp(A[i], B[i], zetta_n[i], R[i], dR[i], epsilon)[0]
        B_temp = B[i] + delta*temp(A[i], B[i], zetta_n[i], R[i], dR[i], epsilon)[1]
        if i == 0:
            A_temp = 0
            B_temp = 0
        A[i+1] = A[i] + (delta/2)*(A_temp-A[i]+temp(A_temp, B_temp, zetta_n[i+1], R[i+1], dR[i+1], epsilon)[0])
        B[i+1] = B[i] + (delta/2)*(B_temp-B[i]+temp(A_temp, B_temp, zetta_n[i+1], R[i+1], dR[i+1], epsilon)[1])
    return [A, B]

for i in range(Rounds):
    epsilon, eigen_vec = finite_differences(A, B)
    R = radial_function(eigen_vec, A)
    dR = der_of_R(R)
    A, B = Runge_Kutta(R, dR, A, B, epsilon)
    print("e: ", epsilon)
    
plt.figure(figsize=(9,9))
plt.plot(zetta_n, R)    
plt.figure(figsize=(9,9))
plt.plot(zetta_n, np.exp(2*A))
plt.plot(zetta_n, np.exp(2*B))