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
import scipy.io as sc 
from sympy import symbols, pprint
import matplotlib.pyplot as plt
from structureSelector import *
from methods.utils.utilities import *
#%%
dataTank = pd.read_csv('data/coupletanks.csv')
u = np.reshape(np.array(dataTank['u']), (1,2401))
y = np.array(dataTank[['tank1', 'tank2']].T)

Na = [12]#[12]
Nb = [2,2]#[2, 2]
level = 1
output = 0
fn = [0,0,0,0,0]
root=True
d = 10

sselector = structureSelector()
ss = sselector.symbolic_regressors(Nb, Na, level, fn, root, d)
print(ss, ss.shape)

vCandidatos = sselector.matrix_candidate(u, y, Nb, Na, level, fn, root, d)
#print(vCandidatos.shape)

pad = max(max(Nb), max(Na))
psi, selected  = sselector.semp(vCandidatos.T, y[output, pad:], 3, 0.00001)
theta = LSM(y[output, pad:], psi)
model = ss[selected]
print(model)
#print(model @ theta)
#%%
slivre = sselector.predict(u, y, theta, ss[selected], Nb, Na, output)

plt.plot(y[output].T, label='Original')
plt.plot(psi @ theta, label='um passo a frente')
plt.plot(slivre, label='Livre')
plt.legend()
plt.show()
#%%
mat_content1 = sc.loadmat("data/ct1e1.mat")
mat_content2 = sc.loadmat("data/ct1e2.mat")

tanque1 = mat_content1['Tanque1']
tanque2 = mat_content2['Tanque2']

t1 = tanque1['time'][0][0]
v1 = tanque1['signals'][0][0]['values'][0][0]

t2 = tanque2['time'][0][0]
v2 = tanque2['signals'][0][0]['values'][0][0]

input = pd.read_csv('data/einput.csv')
t = input['t']
uVal = np.array(input['v']).reshape((1,-1))

plt.plot(t, uVal.T)
plt.plot(t1, v1)
plt.plot(t2, v2)

plt.show()
# %%
v1[v1 < 0] = 0
v2[v2 < 0] = 0
v2[:100] = 0
yVal = np.vstack((v1.T, v2.T))

z = np.zeros(yVal.shape)
valLivre = sselector.predict(uVal, yVal, theta, ss[selected], Nb, Na, output)
yhat = sselector.oneStepForward(uVal, yVal, theta, ss[selected], Nb, Na, output)

plt.plot(yVal[output].T, label='Original')
plt.plot(valLivre, label='Livre')
plt.plot(yhat, label='um passo a frente')
plt.legend()
plt.show()

plt.plot(yVal[output].T - valLivre, label='Livre')
plt.plot(yVal[output].T - yhat, label='um passo a frente')
plt.title("Resíduo")
plt.legend()
plt.show()

pprint(model @ theta)
# %%
def metrics(y, yest):
    residuo1 = y - yest
    mape = round(np.mean(np.abs(residuo1 / (yest + np.finfo(np.float64).eps))), 5)
    print('MSE:', np.mean(np.square(residuo1)), '\nAET:', np.sum(np.abs(residuo1)), '\nMAPE:', str(mape) + '%')
    cc = np.corrcoef(y, yest)
    #print("Correlation pearson:", np.mean(cc))

print("\nSimulação livre")
metrics(yVal[output], valLivre)
print("\nUm passo a frente")
metrics(yVal[output], yhat)
#%%
mat_content1 = sc.loadmat("data/ct1_entry6.mat")
mat_content2 = sc.loadmat("data/ct2_entry6.mat")

tanque1 = mat_content1['Tanque1']
tanque2 = mat_content2['Tanque2']

t1 = tanque1['time'][0][0]
v1 = tanque1['signals'][0][0]['values'][0][0]

t2 = tanque2['time'][0][0]
v2 = tanque2['signals'][0][0]['values'][0][0]

input = pd.read_csv('data/input-test.csv')
t = input['t']
uVal = np.array(input['v']).reshape((1,-1))

plt.plot(t, uVal.T)
plt.plot(t1, v1)
plt.plot(t2, v2)

plt.show()

# %%
v1 = np.ones(uVal.shape)
v2 = np.ones(uVal.shape)
v1[v1 < 0] = 0
v2[v2 < 0] = 0
v2[:100] = 0

yVal = np.vstack((v1, v2))

z = np.zeros(yVal.shape)
valLivre = sselector.predict(uVal, yVal, theta, ss[selected], Nb, Na, output)
yhat = sselector.oneStepForward(uVal, yVal, theta, ss[selected], Nb, Na, output)

plt.plot(uVal.T)
plt.plot(yVal[output].T, label='Original')
plt.plot(valLivre, label='Livre')
plt.plot(yhat, label='um passo a frente')
plt.legend()
plt.show()

plt.plot(yVal[output].T - valLivre, label='Livre')
plt.plot(yVal[output].T - yhat, label='um passo a frente')
plt.title("Resíduo")
plt.legend()
plt.show()

pprint(model @ theta)
