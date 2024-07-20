#created by Jacob Herman

import numpy as np
import matplotlib.pyplot as plt
#global variables-----------------------------------------------------------------

#users input for the code
n = 1000 #interval step
ZETA_MAX = 20 #maximum zeta value
ZETA_MIN = 0 #minimum zeta value
loops = 300 #number of loops the program will iterate through
#---------------------------------------------------------------------------------
#global variables after user input
DELTA =(ZETA_MAX)/(n + 1)
ZETA = np.arange(ZETA_MIN, ZETA_MAX, DELTA) #initializing ZETA values arange(begin, end, stepsize)
N_max = len(ZETA)#set up the size of the matrix's
G = 6.7e-39 #normalized gravity
M_PL = 1 / np.sqrt(G) #mass of plank mass
M = 8.2e10 #if a equals the atomic Bohr radius
a = 1 /(G*M**3)#gravitational bohr radius
R_S = 2*G*M #schwarzschild radius
TOLERANCE = 1.0e-8

#functions------------------------------------------------------------------------


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
    epsilon = lamda/(1 + np.sqrt(1 + (ZETA_S*lamda/2)))#calculates epsilon form lamda
    g00 = np.exp(2*A)
    grr = np.exp(2*B)
    u_tilde = u_bar*np.sqrt(g00/grr)
    u_tilde[0] = 0
    u_tilde[N_max - 1] = 0
    norm = np.sum(grr*u_tilde**2/np.sqrt(g00)) #normalizes u_bar
    #print("norm= ",norm)
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
                
    R_tilde2[0] = R_tilde2[1]
    dR_tilde2[0] = dR_tilde2[1]
    return R_tilde2, dR_tilde2

def temp_slope(ZETA, R_tilde2, dR, A, B, ZETA_S, e):
    '''input single values
    return is single values'''
    #g00 = np.exp(2*A)
    #grr = np.exp(2*B)
    if ZETA == 0:
        dA = 0#keeps our temp values from blowing up to infinity or being undefined
        dB = 0
    else:
        dA = (np.exp(2*B) - 1)/(2*ZETA) - (ZETA_S*ZETA*np.exp(2*B)*R_tilde2)/4 + ((ZETA_S**2)*ZETA/8)*(dR) + ((ZETA_S*ZETA/4)*(1+(ZETA_S*e)/2)**2) * (np.exp(2*B)/np.exp(2*A)) * (R_tilde2)
        dB = -(np.exp(2*B) - 1)/(2*ZETA) + (ZETA_S*ZETA*np.exp(2*B)*R_tilde2)/4 + ((ZETA_S**2)*ZETA/8)*(dR) + ((ZETA_S*ZETA/4)*(1+(ZETA_S*e)/2)**2) * (np.exp(2*B)/np.exp(2*A)) * (R_tilde2)
    
    '''else:
        dA = (grr-1)/(2*ZETA) - (ZETA_S*ZETA*grr*R_tilde2)/4 + ((ZETA_S**2)*ZETA/8)*(dR) + ((ZETA_S*ZETA/4)*(1+(ZETA_S*e)/2)**2) * (grr/g00) * (R_tilde2)
        dB = -(grr-1)/(2*ZETA) + (ZETA_S*ZETA*grr*R_tilde2)/4 + ((ZETA_S**2)*ZETA/8)*(dR) + ((ZETA_S*ZETA/4)*(1+(ZETA_S*e)/2)**2) * (grr/g00) * (R_tilde2)
    '''
    
    return dA, dB

def R_K(A, B, R_tilde2, dR, ZETA_S, e, A_start):
    '''parameters all arrays except e
    return two arrays A's and B's'''
    #A[N_max - 1], B[N_max - 1], M = A_B_end(u_tilde, A, B, ZETA_S)
    A[0] = A_start
    B[0] = 0
    for i in range(N_max -1):
        dA_slope, dB_slope = temp_slope(ZETA[i], R_tilde2[i], dR[i], A[i], B[i], ZETA_S, e)
        if i == 0:
            A_temp = 0#keeps our temp values from blowing up to infinity or being undefined
            B_temp = 0
        else:
            A_temp = A[i] + DELTA*temp_slope(ZETA[i], R_tilde2[i], dR[i], A[i], B[i], ZETA_S, e)[0]
            B_temp = B[i] + DELTA*temp_slope(ZETA[i], R_tilde2[i], dR[i], A[i], B[i], ZETA_S, e)[1]
        
        dA_temp, dB_temp = temp_slope(ZETA[i + 1], R_tilde2[i + 1], dR[i + 1], A_temp, B_temp, ZETA_S, e)
        A[i+1] = A[i] + (DELTA/2)*(dA_slope  + dA_temp)
        B[i+1] = B[i] + (DELTA/2)*(dB_slope + dB_temp)
    return A, B

def secant_meth(x1, x2, fx1, fx2, tol):
    '''
    x1:(float) first initial guess 
    x2:(float) second initial guess
    fx1:(float) A end base of x1
    fx2:(float) B end based of x2
    tol:(float) tolerence for convergents
    '''
    meets_tol = False
    if abs(fx1) < abs(fx2):
        x1_new = x1
    else: 
        x1_new = x2
    x2_new = x2 - fx2*(x2 - x1)/(fx2 - fx1)
    if abs(x2_new - x1_new) <= tol*(abs(x2_new) +abs(x1_new))/2:
        meets_tol = True
    return x1_new, x2_new, meets_tol

def energy_and_mass(ZETA_S, e):
    E_m = 1 + (1/2)*ZETA_S*e
    M_m = (ZETA_S/2)**(1/4)
    return E_m, M_m

def A_B_initialize(ZETA, ZETA_S):
    zeta_0 = 20
    A_new = np.zeros(N_max)
    B_new = np.zeros(N_max)
    g00 = np.ones(N_max)#initializes g00
    grr = np.ones(N_max)#initializes grr
    for i in range(N_max):
        if zeta_0 >= ZETA[i]:
            A_new[i] = (0.5)*np.log((1/4)*(3*np.sqrt(1 - ZETA_S/(zeta_0)) - np.sqrt(1 - (ZETA_S*(ZETA[i])**2)/(zeta_0)**3))**2)
            B_new[i] = (-0.5)*np.log(1 - (ZETA_S*(ZETA[i])**2)/(zeta_0)**3)
        elif zeta_0 < ZETA[i]:
            A_new[i] = (0.5)*np.log(1 - ZETA_S/ZETA[i])
            B_new[i] = (-0.5)*np.log(1 - ZETA_S/ZETA[i])
    #g00 = np.exp(2*A_new)
    #grr = np.exp(2*B_new)
    #h_tilde = ZETA*np.sqrt(np.sqrt(g00/grr))
    return A_new, B_new



#Main Function-------------------------------------------------------

def main():
    zeta_s = [.01, .1, 0.2,0.3,0.4, 0.5, 0.6, 0.7, 0.71, 0.72, 0.725, 0.73, 0.74, 0.741, 0.742, 0.743, 0.744, 0.745, 0.746, 0.747] #how relativistic the function is
    #zeta_s = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.71, 0.715, 0.72, 0.725, 0.73,0.735,0.74]
    #zeta_s = [0.01, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65,0.675, 0.7, 0.701, 0.705, 0.71,0.715, 0.72, 0.725,0.73,0.735,0.74]
    e_array = np.zeros_like(zeta_s)
    E_m = np.zeros_like(zeta_s)
    M_m = np.zeros_like(zeta_s)
    A_0 = np.zeros_like(zeta_s)
    R_tilde2_vals = np.zeros_like(zeta_s)
    print("R_tilde_vals= ", R_tilde2_vals)
    for j, zeta_s_current in enumerate(zeta_s):
        values = []
        ZETA_S = zeta_s[j]
        #A, B, h_tilde = initialize_metric(0.5, ZETA_S)
        if j == 0:
            A, B = A_B_initialize(ZETA, ZETA_S)
            A = np.zeros(N_max)
            B = np.zeros(N_max)
            h_tilde = ZETA*np.sqrt(np.sqrt(np.exp(2*A)/np.exp(2*B)))
            A_start = A[0]
            e = -1
            A_B_sum = 1
        else:
            A = A_end
            B = -A_end

        for i in range(loops):
            prev_e = e
            print("loops= ", i + 1," ZETA_S= ", ZETA_S)
            C, D, F = KG_values(A, B, ZETA_S, h_tilde)
            e, u_bar = KG_solver(C, D, F, A, B, ZETA_S)
            print("Epsilon = ",e)
            meets_tol = False
            A_start1 = A_start + 0.1
            A_start2 = A_start - 0.1
            AB_adjust = 0
            tol = 1e-8
            h_tilde_1 =np.copy(h_tilde)
            h_tilde_2 =np.copy(h_tilde)
            A_1 = np.copy(A)
            A_2 = np.copy(A)
            B_1 = np.copy(B)
            B_2 = np.copy(B)
            while meets_tol == False and AB_adjust < 100:
                AB_adjust += 1
                #h_tilde_1[0] = 0
                h_tilde_1 = ZETA*np.sqrt(np.sqrt(np.exp(2*A_1)/np.exp(2*B_1)))
                R_tilde2_1, dR_1 = gr_R_tildes(A_1, B_1, u_bar, h_tilde_1)#h_tilde
                A_1, B_1 = R_K(A, B, R_tilde2_1, dR_1, ZETA_S, e, A_start1)
                fx1 = A_1[N_max - 1] + B_1[N_max - 1]
                
                #h_tilde_2[0] = 0
                h_tilde_2 = ZETA*np.sqrt(np.sqrt(np.exp(2*A_2)/np.exp(2*B_2)))
                R_tilde2_2, dR_2 = gr_R_tildes(A_2, B_2, u_bar, h_tilde_2)#h_tilde
                A_2, B_2 = R_K(A, B, R_tilde2_2, dR_2, ZETA_S, e, A_start2)
                fx2 = A_2[N_max - 1] + B_2[N_max -  1]
                A_start1, A_start2, meets_tol = secant_meth(A_start1, A_start2, fx1, fx2, tol)
                
            A = np.copy(A_2)
            B = np.copy(B_2)
            A_end = np.copy(A_2)
            B_end = np.copy(B_2)
            A_start = A_start2
            A_0[j]=A_start
            #A_B_sum = A[N_max - 1] + B[N_max - 1]

            if abs(e - prev_e) <= TOLERANCE:
                iter_to_tolerance = i + 1
                print(f"Tolerance met! Took {iter_to_tolerance} iteration(s) for this adjustments")
                break
            h_tilde[0] = 0
            h_tilde = ZETA*np.sqrt(np.sqrt(np.exp(2*A)/np.exp(2*B)))
        #print("R_tilde2_2= ", R_tilde2_2)
        #R_tilde2_vals[j] = R_tilde2_2
        U_abs = abs(u_bar)
        values.append(e)
        array = np.array(values)
        
        e_array[j] = array[len(array) - 1]
        E_m[j], M_m[j] = energy_and_mass(ZETA_S, e_array[j])
        x = A[28] + B[28]
        #print("A + B= ", x)
        
        if ZETA_S == 0.01 or ZETA_S == 0.1 or ZETA_S == 0.2 or ZETA_S == 0.5 or ZETA_S == 0.7 or ZETA_S == 0.74:
            plt.figure(1)
            plt.plot(ZETA, U_abs, label=zeta_s_current, alpha = 0.5, marker = '.')
            plt.xlabel("ZETA")
            plt.xlim(0,25)
            plt.ylabel("U_bar")
            plt.legend()
            plt.grid(True)
            plt.figure(2)
            plt.plot(ZETA, A, label= zeta_s_current)
            plt.plot(ZETA, B, label= zeta_s_current)
            plt.legend()
            plt.xlabel("ZETA")
            plt.xlim(0,20)
            plt.ylabel("arrays")
            plt.grid(True)
        print("ZETA_S= ", zeta_s)
        print("Epsilon array= ",e_array) 
        print("E/M= ", E_m)
        print("M/Mpl= ", M_m)
    #print("epsilon= ", array)
    plt.figure(3)
    #print("A[0]", A_0)
    plt.plot(zeta_s, A_0, alpha = 0.5, marker = '.')
    plt.xlabel("ZETA")
    plt.xlim(0,1)
    plt.ylabel("A[0]")
    #plt.legend()
    plt.grid(True)
    plt.figure(4)
    plt.plot(zeta_s, e_array, alpha = 0.5, marker = '.')
    plt.xlabel("ZETA_s")
    plt.xlim(0,1)
    plt.ylabel("Epsilon")
    #plt.legend()
    plt.grid(True)
    plt.figure(5)
    plt.plot(zeta_s, E_m, alpha = 0.5, marker = '.')
    plt.xlabel("ZETA")
    plt.xlim(0,1)
    plt.ylabel("E/m")
    #plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(6)
    plt.plot(zeta_s, R_tilde2_2, alpha = 0.5, marker = ".")
    plt.xlabel("Zeta")
    plt.ylabel("R_tilde2")
    plt.show
    #print("ZETA_S","Epsilon array","E/M","M/Mpl","epsilon")
    #print(np.vstack(zeta_s),np.vstack(e_array),np.vstack(E_m),np.vstack(M_m),np.vstack(array))
    

if __name__ == "__main__":
    main()