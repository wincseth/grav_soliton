import numpy as np
import matplotlib.pyplot as plt

#global variables-----------------------------------------------------------------

#users input for the code
n = 2000 #interval step
ZETA_MAX = 100 #maximum zeta value
ZETA_MIN = 0 #minimum zeta value
loops = 1 #number of loops the program will iterate through
#---------------------------------------------------------------------------------
#global variables after user input
DELTA =(ZETA_MAX)/(n + 1)
ZETA = np.arange(ZETA_MIN, ZETA_MAX, DELTA) #initializing ZETA values arange(begin, end, stepsize)
N_max = len(ZETA)#set up the size of the matrix's
G = 6.7e-39 #normalized gravity
M_PL = 1 / np.sqrt(G) #mass of plank mass
M = 8.2e10 #if a equals the atomic Bohr radius
#g00 = 1 - zeta/zeta_s #time metric in terms of zeta's
#grr = 1/g00 #radial metric in terms of zeta's
a = 1 /(G*M**3)#gravitational bohr radius
R_S = 2*G*M #schwarzschild radius

#functions------------------------------------------------------------------------

def initialize_metric(ZETA_S):
    '''perameters: 1-d array for A & B
    returns: 1-d array for g00 & grr'''
    A_array = np.zeros(N_max) #initializes the A_array as zeros
    B_array = np.zeros(N_max) #initializes the B_array as zeros
    g00 = np.ones(N_max)#initializes g00
    grr = np.ones(N_max)#initializes grr
    greater_indx = ZETA > ZETA_S
    A_array[greater_indx] = np.log(1 - ZETA_S/ZETA[greater_indx])/2
    B_array[greater_indx] = -np.log(1 - ZETA_S/ZETA[greater_indx])/2
    g00[greater_indx] = 1 -ZETA_S/ZETA[greater_indx]
    grr[greater_indx] = 1/g00[greater_indx]
    h_tilde = ZETA*np.sqrt(np.sqrt(g00/grr))
    return A_array, B_array, h_tilde

def A_B_end(u_tilde, A, B, ZETA_S):
    '''perameters: array
    return: array'''
    g00 = np.exp(2*A)
    grr = np.exp(2*B)
    mu_tilde = np.zeros(N_max)
    dmu_array = np.sqrt(grr/g00)*u_tilde**2
    mu_tilde_end = 0
    for n in range(N_max - 1):
        mu_tilde_end += DELTA*(dmu_array[n] + dmu_array[n + 1])/2
        mu_tilde[n] = mu_tilde_end
    A_end = np.log(1 - ZETA_S*mu_tilde_end/ZETA[N_max - 1])/2
    B_end = -A_end
    return A_end, B_end, mu_tilde

def KG_values(A, B, ZETA_S, h_tilde):
    '''perameters are array's of dimension N_max
    returns are arrays'''
    g00 = np.exp(2*A)
    grr = np.exp(2*B)
    C = np.zeros(N_max)#initalizes our matricies with 0 to calculate the next step
    D = np.zeros(N_max)
    F = np.zeros(N_max)
    Hf_out = np.zeros(N_max)
    for i, zeta in enumerate(ZETA):
        if zeta == ZETA_S:
            print("current ZETA == ZETA_S, no good")
        if i != 0 and i != N_max-1:
            if i != N_max-1:
                D[i] = -np.sqrt((g00[i]*g00[i+1])/(grr[i]*grr[i+1]))/(DELTA**2)
            if i != 0:
                F[i] = -np.sqrt((g00[i]*g00[i-1])/(grr[i]*grr[i-1]))/(DELTA**2)
            h_tilde_frac =(ZETA[i + 1]*np.sqrt(np.sqrt(g00[i + 1]/grr[i + 1])) - 2*ZETA[i]*np.sqrt(np.sqrt(g00[i]/grr[i])) + ZETA[i - 1]*np.sqrt(np.sqrt(g00[i - 1]/grr[i - 1])))/(h_tilde[i]*DELTA**2) 
            Hf_out[i] = h_tilde_frac
        else:
            h_tilde_frac = 0
            Hf_out[i] = h_tilde_frac
        C[i] = ((g00[i]/grr[i])*h_tilde_frac + (4/ZETA_S)*np.exp(A[i])*np.sinh(A[i]) + 2*((g00[i]/grr[i])/(DELTA**2)))#we want c to be in the middle of the tridiagalized matrix
    return C, D, F

def KG_solver(C, D, F, A, B, ZETA_S):
    '''Parameters: returns: epsilon and lmabda as floats, u_bar is a 1-d array
        returns U_bar as an array and epsilon as a schalar '''
    matrix = np.zeros((N_max, N_max))#setting a nxn matrix all zero meshgrid
    for i in range(0, N_max):    
        matrix[i,i] = C[i]
        if i != N_max - 1:
            matrix[i, i + 1] = D[i]
        if i != 0:
            matrix[i, i-1] = F[i]
    e_vals, e_vecs = np.linalg.eig(matrix)#e_vals are lamdas, e_vecs are ubars
    lamda = np.min(e_vals)#finds the minimum e_val and saves as lamda
    u_bar = e_vecs[:, np.argmin(e_vals)]#u_bar is saved as an array of minimum values
    epsilon = lamda/(1 + np.sqrt(1 + (ZETA_S*lamda)/2))#calculates epsilon form lamda
    g00 = np.exp(2*A)
    grr = np.exp(2*B)
    u_tilde = u_bar*np.sqrt(g00/grr)
    u_tilde[0] = 0
    u_tilde[N_max - 1] = 0
    norm = np.sum(grr*u_tilde**2/np.sqrt(g00)) #normalizes u_bar
    u_tilde /= np.sqrt(norm*DELTA)#this takes u_bar and normalizes it to give us a new value of u_bar
    u_bar_new = np.sqrt(grr/g00)*u_tilde
    return epsilon, u_bar_new

def gr_R_tildes(A, B, u_bar, h_tilde):
    g00 = np.exp(2*A)
    grr = np.exp(2*B)
    u_tilde = u_bar*np.sqrt(g00/grr)
    R_tilde2 = np.zeros(N_max)
    dR_tilde2 = np.zeros(N_max)
    for i, zeta in enumerate(ZETA):
        if zeta != 0:
            R_tilde2[i] = (u_bar[i]**2/zeta**2)*np.sqrt(g00[i]/grr[i])
            if i != N_max-1:
                dR_tilde2[i] = (u_tilde[i+1] - u_tilde[i-1])/(2*DELTA*h_tilde[i]) - u_tilde[i]*(h_tilde[i+1]-h_tilde[i-1])/(2*DELTA*h_tilde[i]**2)
                dR_tilde2[i] = dR_tilde2[i]**2
    return R_tilde2, dR_tilde2, u_tilde

def temp_slope(ZETA, R_tilde2, dR, A, B, ZETA_S, e):
    '''input single values
    return is single values'''
    g00 = np.exp(2*A)
    grr = np.exp(2*B)
    if ZETA == 0:
        dA = 0#keeps our temp values from blowing up to infinity or being undefined
        dB = 0
    else:
        dA = (grr-1)/(2*ZETA) - (ZETA_S*ZETA*grr*R_tilde2)/4 + ((ZETA_S**2)*ZETA/8)*(dR) + ((ZETA_S*ZETA/4)*(1+(ZETA_S*e)/2)**2) * (grr/g00) * (R_tilde2)
        dB = -(grr-1)/(2*ZETA) + (ZETA_S*ZETA*grr*R_tilde2)/4 + ((ZETA_S**2)*ZETA/8)*(dR) + ((ZETA_S*ZETA/4)*(1+(ZETA_S*e)/2)**2) * (grr/g00) * (R_tilde2)
    return dA, dB

def R_K(A, B, R_tilde2, dR, ZETA_S, u_tilde, e):
    '''parameters all arrays except e
    return two arrays A's and B's'''
    A[N_max - 1], B[N_max - 1], M = A_B_end(u_tilde, A, B, ZETA_S)
    for i in range(N_max-1, 0, -1):
        dA_slope, dB_slope = temp_slope(ZETA[i], R_tilde2[i], dR[i], A[i], B[i], ZETA_S, e)
        if i == 0:
            A_temp = 0#keeps our temp values from blowing up to infinity or being undefined
            B_temp = 0
        else:
            A_temp = A[i] - DELTA*temp_slope(ZETA[i], R_tilde2[i], dR[i], A[i], B[i], ZETA_S, e)[0]
            B_temp = B[i] - DELTA*temp_slope(ZETA[i], R_tilde2[i], dR[i], A[i], B[i], ZETA_S, e)[1]
        
        dA_temp, dB_temp = temp_slope(ZETA[i - 1], R_tilde2[i - 1], dR[i - 1], A_temp, B_temp, ZETA_S, e)

        A[i-1] = A[i] - (DELTA/2)*(dA_slope  + dA_temp)
        B[i-1] = B[i] - (DELTA/2)*(dB_slope + dB_temp)
    A[0] = 0
    B[0] = 0
    return A, B

def energy_and_mass(ZETA_S, e):
    E_m = 1 + (1/2)*ZETA_S*e
    M_m = (ZETA_S/2)**(1/4)
    return E_m, M_m

def A_B_sphere_app(ZETA, ZETA_S):
    zeta_0 = 0.5
    A_new = np.zeros(N_max)
    B_new = np.zeros(N_max)
    g00 = np.ones(N_max)#initializes g00
    grr = np.ones(N_max)#initializes grr
    for i in range(N_max):
        if zeta_0 >= ZETA[i]:
            A_new[i] = (0.5)*np.log((1/4)*(3*np.sqrt(1 - ZETA_S/(zeta_0)) - np.sqrt(1 - (ZETA_S*(ZETA[i])**2)/(zeta_0)**3))**2)
            B_new[i] = (-0.5)*np.log(1 - (ZETA_S*(ZETA[i])**2)/(zeta_0)**3)
        else:
            A_new[i] = (0.5)*np.log(1 - ZETA_S/ZETA[i])
            B_new[i] = (-0.5)*np.log(1 - ZETA_S/ZETA[i])
    g00 = np.exp(2*A_new)
    grr = np.exp(2*B_new)
    h_tilde = ZETA*np.sqrt(np.sqrt(g00/grr))
    return A_new, B_new, h_tilde