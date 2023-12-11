import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols
import matplotlib.pyplot as plt

def matrix_candidate(u, y, nb, na, level):
    M = []
    size = nb + na
    ni = [size]
    t = np.zeros((level, size))
    t[0, :] = np.ones(size)

    for i in range(1, level):
        k = i + 1
        o = ni[i-1] * (na + nb + k - 1) / k
        ni.append(o)
        s = ni[i-1]
        t[i, 0] = 1
        t[i, 1] = s

        if size > 2:
            for j in range(2, size):
                s = s - t[i-1, j-1]
                t[i, j] = s
    M = []
    H = len(y)
    begin = max(nb, na)
    for i in range(begin, H):
        l = []
        l1 = []
        l1 = np.hstack((np.flipud(y[i-nb:i]), np.flipud(u[i-na:i])))
        li = l1
        l.append(l1)
        aux = l1
        for j in range(level-1):
            lx = []
            k = 0

            for v in range(size):
                k += t[j, v]
                lx.extend(np.array(aux[int(k-1):]) * li[v])

            l1 = np.hstack((l1, lx))
            aux = lx

        M.append(l1)
    phi = np.array(M)
    ones = np.ones((phi.shape[0], 1))
    phi1 = np.hstack((ones, phi))
    return phi1

def calculeg(phi, y, nb, na):
    s = max(nb, na)
    yy = y[s:]

    M = phi.shape
    alpha = np.dot(yy.T, yy)

    g = np.zeros(M[1], np.float64)
    err = np.zeros(M[1], np.float64)

    for i in range(M[1]):
        pp = np.dot(phi[:, i].T, phi[:, i])
        g[i] = np.dot(yy.T, phi[:, i]) / pp
        err[i] = (np.dot(yy.T, phi[:, i])**2)/ (pp * alpha)#(g[i]**2) * pp / alpha

    return g, err

def symbolic_candidates(nb, na, level):
  M = []
  size = na + nb
  ni = [size]
  t = np.zeros((level, size))
  t[0,:] = np.ones((1, size))
  for i in range(1, level):
    o = ni[i-1] * (na + nb + (i+1) - 1) / (i+1)
    ni.append(o)
    s = ni[i-1]
    t[i, 0] = 1
    t[i, 1] = s
    if size > 2:
      for j in range(2, size):
        s = s - t[i-1, j-1]
        t[i, j] = s
  
  l = []
  l1 = []
  ry = sp.zeros(1, nb)
  for i in range(nb):
    ry[i] = sp.symbols('Y'+str(i+1))

  ru = sp.zeros(1, na)
  for i in range(na):
    ru[i] = sp.symbols('U'+str(i+1))

  l1 = list(np.hstack((ry, ru))[0])
  li = l1.copy()
  l.append(l1)
  aux = l1.copy()
  for j in range(level-1):
    lx = []
    k = 0
    for i in range(size):
      k += t[j, i]
      lx = lx + (list(np.array(aux[int(k-1):]) * li[i]))
    l1 = np.hstack((l1, lx))
    aux = lx
  M = np.hstack((sp.symbols('1'), l1))#l1
  return M


def oneStepForward(u, y, theta, model, nb, na):
  yest = np.zeros(y.shape)
  d = max(na, nb)
  yest[:d] = y[:d]
  sym = symbols('Y1:{}'.format(nb+1)) + symbols('U1:{}'.format(na+1))
  for k in range(d, y.shape[0]):
    num = np.hstack((np.flip(y[k-nb:k]), np.flip(u[k-na:k])))
    dicionario = dict(zip(sym, num))
    dicionario['1'] = 1
    aux = np.array([m.evalf(subs=dicionario) for m in model])
    yest[k] = aux @ theta
  return yest

def LSM(y, fi):
  fi = np.array(fi)
  Y = y.T.copy()
  #print(fi.shape, Y.shape)
  theta = np.linalg.inv(fi.T @ fi) @ fi.T @ Y
  return theta

def predict(u, y, theta, model, nb, na):
  yest = np.zeros(y.shape)
  d = max(na, nb)
  yest[:d] = y[:d]
  sym = symbols('Y1:{}'.format(nb+1)) + symbols('U1:{}'.format(na+1))

  for k in range(d, y.shape[0]):
    num = np.hstack((np.flip(yest[k-nb:k]), np.flip(u[k-na:k])))
    dicionario = dict(zip(sym, num))
    dicionario['1'] = 1
    aux = np.array([m.evalf(subs=dicionario) for m in model])
    #print(y[k], aux @ theta, aux, phi[k-d])
    #print(aux.shape, theta.shape)

    yest[k] = aux @ theta
  return yest

def metric(u, y, theta, model, nb, na):
  print('Simulação livre')
  yest = predict(u, y, theta, model, nb, na)
  residuo1 = y - yest
  mape = round(np.mean(np.abs(residuo1 / (yest + np.finfo(np.float64).eps))), 5)
  print('MSE:', np.mean(np.square(residuo1)), '\nAET:', np.sum(np.abs(residuo1)), '\nMAPE:', str(mape) + '%')

  print('\nSimulação um passo a frente')
  yhat = oneStepForward(u, y, theta, model, nb, na)
  residuo2 = y - yhat
  mape = round(np.mean(np.abs(residuo2 / (yhat + np.finfo(np.float64).eps))), 5)
  print('MSE:', np.mean(np.square(residuo2)), '\nAET:', np.sum(np.abs(residuo2)), '\nMAPE:', str(mape) + '%')

  return yest, yhat, residuo1, residuo2

def plotting(y, est, yVal, yest, yhat, pad, r1, r2):

  f, ax = plt.subplots(2, 2, figsize=(15,8))
  ax[0][0].plot(y[pad:], label='Original')
  ax[0][0].plot(est, label='Estimado')
  ax[0][0].legend()
  ax[0][0].set_title('Estimação')

  ax[0][1].plot(yVal, label='Original')
  ax[0][1].plot(yest, label='Livre')
  ax[0][1].legend()
  ax[0][1].set_title('Validação')

  ax[1][0].plot(yVal, label='Original')
  ax[1][0].plot(yhat, label='1 passo a frente')
  ax[1][0].legend()
  ax[1][0].set_title('Validação')

  ax[1][1].plot(r1, label='Livre')
  ax[1][1].plot(r2, label='1 passo a frente')
  ax[1][1].legend()
  ax[1][1].set_title('Resíduos')
  plt.show()