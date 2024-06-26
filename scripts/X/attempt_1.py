#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:36:13 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt


n_int = 500
zetta_max = 180
zetta_sn = [0.01, .1, .2, .5, 1]
colors = ["red", 'orange', 'yellow', 'green', 'blue']
delta = zetta_max/(n_int+1)
zetta_n = np.arange(0, zetta_max, delta)
Rounds = 10
G = 6.7*10**(-39) #normalized gravity
M_PL = 1 / np.sqrt(G) #mass of plank mass
M = 8.2*10**10
a = 1 /(G*M**3)
n = len(zetta_n)

#Finite differences attempt 1
def finite_differences(A, B):
    goo = np.exp(2*A) #g_00 guess
    grr = np.exp(2*B) #g_rr guess
    goo_approx = 2*np.exp(A)*np.sinh(A) #approx. used for g_00(1-1/g_00)
    matrix = np.zeros((n, n)) #creating a matrix of 0s
    for i in range(n): #adding the values into 0 for all the equations from 0 to n
        if i == 0:
            H = 0
        else:
            H = (zetta_s**2/((4*zetta_n[i]**2)*(zetta_n[i]**2-zetta_s**2)))
        C = -(goo[i]/grr[i])*H+(2/zetta_s)*goo_approx[i]+2*(goo[i]/grr[i])/delta**2 #first value
        matrix[i, i] = C
        if i < n-1:
            D = -np.sqrt(goo[i]/grr[i])*np.sqrt(goo[i+1]/grr[i+1])/(delta**2) #second value
            matrix[i, i+1] = D
        if i > 0:
            E = -np.sqrt(goo[i]/grr[i])*np.sqrt(goo[i-1]/grr[i-1])/(delta**2)
            matrix[i, i-1] = E
    eigenvalues, eigenvectors = np.linalg.eig(matrix) #getting the eigenvalues and eigenvectors
    #print(N, eigenvalues[N])
    epsilon = eigenvalues/(1+np.sqrt(1+zetta_s*eigenvalues/2))
    N = np.argmin(epsilon)
    u_bar = eigenvectors[:, N]
    u_bar = np.sqrt(goo/grr)*u_bar
    norm = sum(np.sqrt(grr)*u_bar**2*delta)
    u_bar /= np.sqrt(norm)
    u_bar[0] = 0
    print(sum(u_bar))
    
    return [min(epsilon), u_bar]

def radial_function(u, A, zetta):
    R = np.zeros_like(u)
    use = np.where(zetta != 0)
    R[use] = np.sqrt(np.sqrt(np.exp(2*A[use])/np.exp(2*B[use])))*u[use]/(zetta_n[use])
    return R

def der_of_R(R):
    dR = np.zeros(len(R))
    for i in range(len(R)):
        if i == n-1 or i == 0:
            dR[i]=0
            continue
        dR[i] = (R[i+1]-R[i-1])/(2*delta)
    return(dR)

def temp(A, B, R, dR, epsilon, zetta):
    goo = np.exp(2*A)
    grr = np.exp(2*B)
    common = (zetta_s*zetta/4)*(1+zetta_s*epsilon/2)**2*(grr/goo)*R**2
    A_temp =(1/(2*zetta))*(grr-1)-((zetta_s*zetta/4)*grr*(R**2))+(zetta_s**2*zetta/8)*dR**2+common
    B_temp =-(1/(2*zetta))*(grr-1)+(zetta_s*zetta/4)*grr*R**2+(zetta_s**2*zetta/8)*dR**2+common
    return [A_temp, B_temp]

#now, attempting RK
def Runge_Kutta(R, dR, epsilon):
    A = np.zeros(len(R))
    B = np.zeros(len(R))
    for i in range(len(R)-1):
        if i == 0:
            A[i] = 0
            B[i] = 0
            continue
        
        A_temp = A[i] + delta*temp(A[i], B[i], R[i], dR[i], epsilon, zetta_n[i])[0]
        B_temp = B[i] + delta*temp(A[i], B[i], R[i], dR[i], epsilon, zetta_n[i])[1]
        
        A[i+1] = A[i] + (delta/2)*(+temp(A_temp, B_temp, R[i+1], dR[i+1], epsilon, zetta_n[i+1])[0])
        B[i+1] = B[i] + (delta/2)*(+temp(A_temp, B_temp, R[i+1], dR[i+1], epsilon, zetta_n[i+1])[1])
    return [A, B]

def U_final(g00, grr,u):
    U_final = (u/np.sqrt(a))*np.sqrt(g00/grr)
    return U_final

plt.figure(figsize=(9,9))
epsilons = np.zeros_like(zetta_sn)
A = np.zeros_like(zetta_n)
B = np.zeros_like(zetta_n)
A_final = np.zeros((len(zetta_sn), len(A)))
B_final = np.zeros((len(zetta_sn), len(A)))
for j in range(len(zetta_sn)):
    zetta_s = zetta_sn[j]
    for i in range(n):
        if zetta_n[i] <= zetta_s:
            A[i] = 0
            B[i] = 0
        if zetta_n[i] > zetta_s:
            A[i] = np.log(1-(zetta_s/zetta_n[i]))/2
            B[i] = -A[i]
    for i in range(Rounds):
        epsilon, eigen_vec = finite_differences(A, B)
        R = radial_function(eigen_vec, A, zetta_n)
        dR = der_of_R(R)
        A, B = Runge_Kutta(R, dR, epsilon)
        print("e: ", epsilon)
    U = U_final(np.exp(2*A), np.exp(2*B), eigen_vec)
    epsilons[j] = epsilon
    A_final[j] = A
    B_final[j] = B
    plt.plot(zetta_n, abs(eigen_vec), label=f'zeta_s value of {zetta_sn[j]}')
ax = plt.gca()
ax.set_xlim([0, 25])
plt.legend()
print(epsilons)
print(1+epsilons*zetta_s/2)
    
  
plt.figure(figsize=(9,9))
for j in range(len(zetta_sn)):
    plt.plot(zetta_n, np.exp(2*A_final[j]), label=f'zeta_s value of {zetta_sn[j]}')
plt.legend()

plt.figure(figsize=(9,9))
for j in range(len(zetta_sn)):
    plt.plot(zetta_n, np.exp(2*B_final[j]), label=f'zeta_s value of {zetta_sn[j]}')
plt.legend()