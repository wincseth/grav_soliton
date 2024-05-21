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
delta = 0.01
n = 20
zetta = n*delta
second_der = 0

goo = np.exp(2*A)
grr = np.exp(2*B)

goo_approx = 2*np.exp(A)*np.sinh(A)

for value in range(n):
    second_der += 