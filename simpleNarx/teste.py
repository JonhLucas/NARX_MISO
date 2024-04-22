#%%
import numpy as np
import pandas as pd
import sympy as sp
import scipy.io as sc 
import scipy.signal as signal
from sympy import symbols, pprint
import matplotlib.pyplot as plt
import plotly.express as px

def integrate(p, ts=0.1):
    if p.shape[1] > p.shape[0]:
        p = p.T
    r = np.zeros(p.shape)
    r[0] = p[0] * ts
    for i in range(1, p.shape[0]):
        r[i] = r[i-1] + p[i] * ts
    return r

ballbeam  = np.loadtxt('../data/ballbeam.dat')
print(ballbeam.shape)
part = 490#ballbeam.shape[0] // 2#700

u = ballbeam[:part, 0].reshape((1,-1))
y = ballbeam[:part, 1].reshape((1,-1))

wn = 0.1
b1, a1 = signal.butter(4, wn, 'low')

filtered = signal.filtfilt(b1, a1, y, padlen=100)

dy = np.zeros(filtered.shape)
dy[:, 1:] = (filtered[0, 1:] - filtered[0, :-1]) / 0.1

t = np.arange(0, part/10, 0.1)

#u -= u[0,0]
wn = 0.2
b1, a1 = signal.butter(5, wn, 'low')
ufiltered = signal.filtfilt(b1, a1, u, padlen=100)
#ufiltered[0, :15] = 0
U = integrate(ufiltered.T, 0.5).T

w = 3

du = np.zeros(filtered.shape)
du[:, 1:] = (ufiltered[0, 1:] - ufiltered[0, :-1]) / 0.1
g = filtered * du**2 * 1e2

plt.axhline(y=0, color='red')
plt.plot(t, dy.T)
plt.plot(t, U.T, '-', label='derivada')
#plt.plot(t[w:], U[0, :-w].T, '-o', label='entrada')
plt.axhline(y=0, color='red')
plt.legend()
plt.show()
#%%
'''plt.axhline(y=0, color='red')
plt.plot(t, g.T)
plt.plot(t, ufiltered.T, '-', label='derivada')
#plt.plot(t[w:], U[0, :-w].T, '-o', label='entrada')
plt.axhline(y=0, color='red')
plt.legend()
plt.show()
#%%
plt.plot(integrate(0.5 * g.T + u.T))
plt.show()'''
#%%
d2y = np.zeros(filtered.shape)
d2y[:, 1:] = (dy[0, 1:] - dy[0, :-1]) / 0.1
plt.plot(d2y.T)
plt.plot(u.T * 5)
plt.show()
#%%
d3y = np.zeros(filtered.shape)
d3y[:, 1:] = (d2y[0, 1:] - d2y[0, :-1]) / 0.1
du = np.zeros(filtered.shape)
du[:, 1:] = (ufiltered[0, 1:] - ufiltered[0, :-1]) / 0.1
plt.axhline(y=0, color='red')
plt.plot(d3y.T)
plt.plot(du.T * 5)
plt.show()