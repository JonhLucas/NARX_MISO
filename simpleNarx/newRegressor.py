#%%
import numpy as np
import pandas as pd
import sympy as sp
import scipy.io as sc 
from sympy import symbols, pprint
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("..")

from structureSelector import *
from methods.utils.utilities import *

dataTank = pd.read_csv('../data/coupletanks.csv')
u = np.reshape(np.array(dataTank['u']), (1,-1))
y = np.array(dataTank[['tank1', 'tank2']].T)

#%%
#Selecione o tanque 
output = 0  # 0 ou 1

num = [3, 5]
params = []
params.append({'nb':[4,2],'na':[10], 'level':1, 'nonlinear':[0, 0,0,0,0], 'root':True, 'delay':8, 'diff':False})
params.append({'nb':[0,2],'na':[4], 'level':2, 'nonlinear':[0,0,0,0,0], 'root':True, 'delay':0, 'diff':False})

sselector = structureSelector()
ss = sselector.symbolic_regressors(**params[output])
vCandidatos = sselector.matrix_candidate(u, y, **params[output], dt=0.1)

print(ss)
print(len(ss), vCandidatos.shape[0])

# %%
labs = ['1', '2', '3', '4', '5', '6']
plt.plot(vCandidatos[-8:-4].T, label=labs[-6:-2])
plt.legend()
plt.show()