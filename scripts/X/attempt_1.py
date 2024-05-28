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
n = 100
N = 1
delta = 1/(n+1)
zetta = n*delta
zetta_n = np.arange(0, zetta_s, zetta_s/n)
x = np.linspace(-100, 100, n)  
y = np.linspace(-100, 100, n)
X, Y = np.meshgrid(x, y)


#Finite differences attempt 1
def finite_differences(A, B, N):
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
    eig_vec = eigenvectors[:, N].reshape((n, 1))
    eig_vec = np.tile(eig_vec, (1, n)) / np.sqrt(np.abs(X) + 1e-10) #reshaping and normalizing
    eigen_vec = eigenvectors[:, N]
    epsilon = eigenvalues/(1+np.sqrt(1+zetta*eigenvalues/2))
    return [epsilon, eig_vec, eigen_vec]

def radial_function(u):
    R = np.sqrt(np.exp(2*A))*u / zetta
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

epsilon_n = finite_differences(A, B, N)[0]
epsilon = epsilon_n[N]

#now, attempting RK
def Runge_Kutta(R, dR, epsilon):
    A = np.zeros(len(R))
    B = np.zeros(len(R))
    for i in range(len(R)-1):
        print(A[i], B[i], zetta_n[i], R[i], dR[i], epsilon)
        A_temp = A[i] + delta*temp(A[i], B[i], zetta_n[i], R[i], dR[i], epsilon)[0]
        B_temp = B[i] + delta*temp(A[i], B[i], zetta_n[i], R[i], dR[i], epsilon)[1]
        print(A_temp, B_temp)
        A[i+1] = A[i] + (delta/2)*temp(A[i], B[i], zetta_n[i], R[i], dR[i], epsilon)[0]+temp(A_temp, B_temp, zetta_n[i+1], R[i+1], dR[i+1], epsilon)[0]
        B[i+1] = B[i] + (delta/2)*temp(A[i], B[i], zetta_n[i], R[i], dR[i], epsilon)[1]+temp(A_temp, B_temp, zetta_n[i+1], R[i+1], dR[i+1], epsilon)[1]
    return [A, B]


print(epsilon)
R = radial_function(finite_differences(A, B, N)[2])
print(R)
dR = der_of_R(R)
print(dR)
v = Runge_Kutta(R, dR, epsilon)
print(v)
plt.figure(figsize=(9,9))
plt.contourf(X, Y, finite_differences(A, B, N)[1]**2, n) #plotting to visualize probability density
plt.savefig('Probability density')
plt.figure(figsize=(9,7))
plt.plot(x, R)
plt.savefig('R Function')
plt.figure(figsize=(9,7))
plt.plot(x, dR)
plt.savefig('Derivative of R')