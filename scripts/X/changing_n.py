#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:52:10 2024

@author: xaviblast123
"""

import numpy as np
import matplotlib.pyplot as plt
from attempt_2 import function_of_main

n = np.linspace(100, 1500, 15)
a_array = np.zeros_like(n)
b_array = np.zeros_like(n)
for i in range(len(n)):
    a_array[i], b_array[i] = function_of_main(n, 100, 1, 25)
    
print(a_array, b_array)