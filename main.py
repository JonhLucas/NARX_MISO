#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 19:38:25 2023

@author: jonhlucas
"""
#%%
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, pprint
import matplotlib.pyplot as plt
from structureSelector import *
from methods.utils.utilities import *

dataTank = pd.read_csv('coupletanks.csv')
u = np.reshape(np.array(dataTank['u']), (1,2401))
y = np.array(dataTank[['tank1', 'tank2']].T)

Na = [2]
Nb = [2,0]#[2, 2]
level = 1
output = 0
fn = [0,0,0,0,0]

sselector = structureSelector()
ss = sselector.symbolic_regressors(Nb, Na, level, fn, True)
print(ss, ss.shape)
vCandidatos = sselector.matrix_candidate(u, y, Nb, Na, level, fn, root=True)
print(vCandidatos.shape)
#%%
pad = max(max(Nb), max(Na))
psi, selected  = sselector.semp(vCandidatos.T, y[output, pad:], 3, 0.00001)
theta = LSM(y[output, pad:], psi)
print(psi.shape, ss[selected], theta)
#%%
slivre = sselector.predict(u, y, theta, ss[selected], Nb, Na, output)
#%%
plt.plot(y[output].T)
'''plt.plot(psi @ theta)'''
plt.plot(slivre)
plt.show()
#%%
plt.plot((y[0,:] * y[0,:]).T)
plt.plot(vCandidatos[11,:].T)
plt.show()

 