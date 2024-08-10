"""
Created on Wed Jul  3 11:17:06 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as r

epsilons_seth=[-0.3488294977377729, -0.3480796336225954, -0.34776111237120133, -0.34741531230096007]
epsilons_xavier = [-0.34874277862134573, -0.34817497561046995, -0.3476974450701045, -0.3474959585918155]
A_0_seth = [-0.3174721475829741,-0.31657573424729224,-0.3161941817170664,-0.3157816722886458]
A_0_xavier = [-0.317369134168473,-0.3166897579035516,-0.3161188815299597,-0.3158781468863246]
delta = [0.000594, 0.000495, 0.000371, 0.000371, 0.000297]
n = [1250, 1500, 2000, 2500]
#Main Function-------------------------------------------------------------------------
def main():
    

    #Plot of U_bars
    plt.figure(figsize=(9,9))
    plt.plot(epsilons_seth, n, marker='o', label=f"Seth's epsilon\n")
    plt.plot(epsilons_xavier, n, marker='o', label = f"Xavier's epsilon\n")
    plt.legend()
    plt.xlabel('Epsilons')
    plt.ylabel('Interval steps')
    plt.title(f'Interval steps vs Epsilons at $\zeta_s$ = 0.7427')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'images/U_bars_{name}.png')
    
    plt.figure(figsize=(9,9))
    plt.plot(A_0_seth, n, marker='o', label=f"Seth's A_0\n")
    plt.plot(A_0_xavier, n, marker='o', label = f"Xavier's A_0\n")
    plt.legend()
    plt.ylabel('Interval steps')
    plt.xlabel('A_0')
    plt.title(f'Interval steps vs A_0 at $\zeta_s$ = 0.7427')
    plt.legend()
    plt.grid(True)
    #plt.savefig(f'images/U_bars_{name}.png')

    
    plt.show()
_=main()