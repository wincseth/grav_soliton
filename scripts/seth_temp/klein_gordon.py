# ZETA_S = 0.01, del = 0.01, n = 20, A = B = 0

import numpy as np
import matplotlib.pyplot as plt

ZETA_S = 0.01 #schwarchild radius and bohr radius ratio
del_step = 0.01 #step size
n = 20
A, B = 0 #related to metric
g_00 = np.exp(2*A)
g_rr = np.exp(2*B)



