import numpy as np
import sympy as sp
from .utils.utilities import LSM

def semp(psi, y, ni, rho = 0.00001):
  idx = np.arange(0, psi.shape[1])
  selected = []
  #print(idx)
  P = np.array([])
  Q = psi.copy()

  t = LSM(y, psi)
  Jold = np.inf#np.mean(np.square(y - (psi @ t)))

  #rho = 0.00001
  for i in range(ni):
    J = np.array([])

    for j in range(Q.shape[1]):
      q = Q[:,j].reshape((-1,1))
      if i == 0:
        p = np.append(P, q).reshape((-1, 1))
      else:
        p = np.hstack((P,q))
      #print('oj:', p.shape, q.shape, P.shape)
      theta = (np.linalg.inv(p.T @ p) @ p.T) @ y
      J = np.append(J, np.mean(np.square(y - (p @ theta))))
    l = np.argmin(J)
    #print("New J:", J[l], l)
    if J[l] < Jold and np.abs(J[l] - Jold) > rho:
      if P.shape[0] == 0:
        P = np.append(P, Q[:, l]).reshape((-1,1))
      else:
        P = np.hstack((P, Q[:, l].reshape((-1,1))))
      Q = np.delete(Q, l, 1)
      selected.append(idx[l])
      idx = np.delete(idx, l)
      #print(idx, selected)
    else:
      #print("Encerrando", Jold, J[l], np.abs(J[l] - Jold) > rho)
      return P, selected

    #print("Prunning", P.shape, Q.shape)
    flag = True
    while P.shape[1] > 1 and flag:
      Jp = np.array([])
      for k in range(P.shape[1]):
        R = np.delete(P, k, 1)
        theta = (np.linalg.inv(R.T @ R) @ R.T) @ y
        Jp = np.append(Jp, np.mean(np.square(y - (R @ theta))))
      m = np.argmin(Jp)
      #print('Jp:', Jp.shape, P.shape)
      if Jp[m] < Jold:
        #print(m, len(selected), P.shape[1], Jp[m], Jold)
        #print('Deletando:', m, selected[m])
        P = np.delete(P, m, 1)
        selected.pop(m)
        continue
      else:
        flag = False #revisar
    #atualizando Jold
    theta = (np.linalg.inv(P.T @ P) @ P.T) @ y
    Jold = np.mean(np.square(y - (P @ theta)))#J[l]
    #print('----- Jold', Jold, P.shape, '\n')
  #print('Fim:', P.shape)
  return P, selected