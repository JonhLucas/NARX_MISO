import numpy as np
import sympy as sp
from .utils.utilities import *

#%%frols

def frols(nb, na, u, y, level, num, verbose=False):
  #passo 1
  l = []
  phi = matrix_candidate(u, y, nb, na, level)
  gm1, err1 = calculeg(phi, y, nb, na)
  l1 = np.argmax(err1)

  if verbose:
    print("l1:", l1, "error:", err1[l1])
    print('ESR', 1 - err1[l1])
  q = np.zeros(phi.shape);
  g = np.zeros((phi.shape[1]))
  err = np.zeros((phi.shape[1]))
  #error_reduction_ratio(phi, y, 15)
  q[:, 0] = phi[:, l1]
  g[0] = gm1[l1]
  err[0] = err1[l1]
  #recuperação de indice
  indice_backup = np.arange(0, phi.shape[1])
  l.append(l1)

  Ds = phi[:, l1].reshape((phi.shape[0],1))
  dd = np.delete(phi, l1, axis=1)
  indices = np.delete(indice_backup, l1)

  #Passo s
  for t in range(num - 1):
    qsm = []
    for m in range(dd.shape[1]):
      pm = dd[:, m]
      s = np.zeros((dd.shape[0]))
      for i in range(Ds.shape[1]):#
        qr = Ds[:, i]
        qs = qr * (pm.T @ qr) / (qr.T @ qr)
        s = s + qs
      qsm.append(pm-s)

    qsm = np.array(qsm).T
    gm, errs = calculeg(qsm, y, nb, na)
    ls = np.argmax(errs)
    err[1 + t] = errs[ls]


    ps = qsm[:, ls].reshape((qsm.shape[0], 1))
    Ds = np.hstack((Ds, ps))

    l.append(indices[ls])
    dd = np.delete(phi, l, axis=1)
    indices = np.delete(indice_backup, l)
    if verbose:
      print("l"+str(2 + t)+":", ls, "error:", errs[ls])
      print('ESR', 1 - sum(err[:t+2]), err[:t+2])
  return l, err[:num], np.array(phi[:, l])

