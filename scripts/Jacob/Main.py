#created by Jacob Herman

import numpy as np
import matplotlib.pyplot as plt
#import functions-----------------------------------------------------------------
from functions import initialize_metric
from functions import A_B_end
from functions import KG_values
from functions import KG_solver
from functions import gr_R_tildes
from functions import R_K
from functions import energy_and_mass
from functions import A_B_sphere_app
from functions import loops
from functions import ZETA
#Main Function-------------------------------------------------------

def main():
    #zeta_s = [.01, .1, .2, 0.5, 1] #how relativistic the function is
    zeta_s = [0.01, 0.1, 0.2, 0.3, 0.4]
    e_array = np.zeros_like(zeta_s)
    E_m = np.zeros_like(zeta_s)
    M_m = np.zeros_like(zeta_s)
    
    for j, zetta_s_current in enumerate(zeta_s):
        values = []
        ZETA_S = zeta_s[j]
        #A, B, h_tilde = initialize_metric(ZETA_S)
        A, B, h_tilde = A_B_sphere_app(ZETA, ZETA_S)
        for i in range(loops):
            print("loops= ", i + 1," ZETA_S= ", ZETA_S)
            C, D, F = KG_values(A, B, ZETA_S, h_tilde)
            e, u_bar = KG_solver(C, D, F, A, B, ZETA_S)
        
            R_tilde2, dR, u_tilde = gr_R_tildes(A, B, u_bar, h_tilde)
            A_end, B_end, mu_tilde = A_B_end(u_tilde, A, B, ZETA_S)
            A, B = R_K(A, B, R_tilde2, dR, ZETA_S, u_tilde, e)
            h_tilde[0] = 0
            g00 = np.exp(2*A)
            grr = np.exp(2*B)
            h_tilde = ZETA*np.sqrt(np.sqrt(g00/grr))
            U_abs = abs(u_bar)
        g00 = np.exp(2*A)
        grr = np.exp(2*B)
        values.append(e)
        array = np.array(values)
        e_array[j] = array[len(array) - 1]
        E_m[j], M_m[j] = energy_and_mass(ZETA_S, e_array[j])
        print("ZETA_S= ", zeta_s)
        print("Epsilon array= ",e_array) 
        print("E/M= ", E_m)
        print("M/Mpl= ", M_m)
        print("U= ", U_abs)
        print("epsilon= ",array)
        print("lowest epsilon= ", array[len(array) - 1])
        plt.figure(1)
        plt.plot(ZETA, U_abs, label=zetta_s_current, alpha = 0.5, marker = '.')
        plt.xlabel("ZETA")
        plt.xlim(0,30)
        plt.ylabel("U_bar")
        plt.legend()
        plt.grid(True)
        plt.figure(2)
        plt.subplot(211)
        plt.plot(ZETA, A, label=zetta_s_current)
        plt.xlabel("ZETA")
        plt.xlim(0,30)
        plt.ylabel("A_array")
        plt.grid(True)
        plt.subplot(212)
        plt.plot(ZETA, B, label=zetta_s_current)
        plt.xlabel("ZETA")
        plt.xlim(0,30)
        plt.ylabel("B_array")
        plt.grid(True)
        '''
        plt.figure(3)
        plt.plot(ZETA, R_tilde2, label=zetta_s_current)
        plt.xlabel("ZETA")
        plt.xlim(0,30)
        plt.ylabel("R_TILDE")
        plt.legend()
        plt.grid(True)
        plt.figure(4)
        plt.plot(ZETA, mu_tilde, label=zetta_s_current)
        plt.xlabel("ZETA")
        plt.xlim(0,60)
        plt.ylabel("mu_tilde")
        plt.legend()
        plt.grid(True)
        plt.figure(5)
        plt.plot(ZETA, A_new, label=zetta_s_current)
        plt.plot(ZETA, B_new, label=zetta_s_current)
        plt.xlabel("ZETA")
        plt.xlim(0,20)
        plt.legend()
        plt.ylabel("A_B_new")
        plt.grid(True)'''
    plt.show()
if __name__ == "__main__":
    main()