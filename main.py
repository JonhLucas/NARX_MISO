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
Nb = [2,2]#[2, 2]
level = 1
output = 1
fn = [0,0,0,0,0]

sselector = structureSelector()
ss = sselector.symbolic_regressors(Nb, Na, level, fn, True)
print(ss, ss.shape)
vCandidatos = sselector.matrix_candidate(u, y, Nb, Na, level, fn, root=True)
print(vCandidatos.shape)
#%%
pad = max(max(Nb), max(Na))
psi, selected  = sselector.semp(vCandidatos.T, y[output, pad:], 4, 0.00001)
theta = LSM(y[output, pad:], psi)
print(psi.shape, ss[selected], theta)
#%%
slivre = sselector.predict(u, y, theta, ss[selected], Nb, Na, output)
#%%
plt.plot(y[output].T, label='Original')
plt.plot(psi @ theta, label='um passo a frente')
plt.plot(slivre, label='Livre')
plt.legend()
plt.show()
#%%

 