import numpy as np
import matplotlib.pyplot as plt

A = 0 #initial condition for time metric  
B = 0 #initial condition for radial matric 
n = 20 # steps
DELTA = 1 / (n+1) # finite difference?
ZETA = n * DELTA
ZETA_S = 0.01 #initialized from some eigenvalues
G = 6.7*10**(-39) #normalized gravity
M_PL = 1 / np.sqrt(G) #mass of plank mass
m = 8.2*10**10 #if a equals the atomic Bohr radius
a = 1 / (G*m**3)#gravitational bohr radius
r_s = 2*G*m #schwarzschild radius

def finite_diff(A, B):
    g00 = np.exp(2*A) #time metric
    grr = np.exp(2*B) #radial metric
    H = (ZETA_S**2)/(4*(ZETA**2)*(ZETA**2 - ZETA_S**2))  # this is h_~"/h_~ = H
    C = (-(g00/grr)*H + (4/ZETA_S)*np.exp(A)*np.sinh(A) + 2*((g00/grr)/(DELTA**2)))#matrix component 
    D = -(g00/grr)/(DELTA**2)#matrix component
    matrix = np.array([[C,D],[D,C]])
    e_vals, e_vec = np.linalg.eig(matrix)
    
    epsilon = e_vals/(1+np.sqrt(1 + (ZETA_S*e_vals)/2))
    
    return [epsilon, e_vec]
x,y = finite_diff(A,B)#x= e_vals, y= e_vect
#print("matrix", finite_diff(A, B))#prints e_vals and e_vectors
print("e_vals", x)#prints e_vals
print("e_vec", y)#prints e_vectors

#Runge-Kutta method

