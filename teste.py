#%%
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, pprint
import matplotlib.pyplot as plt
'''from methods.frols import *
from methods.semp import *'''
from methods.utils.utilities import *


def symbolic_candidates(nb, na, level, root=False):
  M = []
  nb = np.array(nb)
  na = np.array(na)
  ny = np.sum(nb)
  nx = np.sum(na)
  #print(ny, nx)
  
  size = nx + ny
  ni = [size]
  t = np.zeros((level, size))
  t[0,:] = np.ones((1, size))
  for i in range(1, level):
    o = ni[i-1] * (nx + ny + (i+1) - 1) / (i+1)
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
  ry = sp.zeros(1, ny)
  p = 0
  for i in range(nb.shape[0]):
    for j in range(0, nb[i]):
      ry[p+j] = sp.symbols("Y"+str(i+1)+"."+str(j+1))
    p += nb[i]

  ru = sp.zeros(1, nx)
  e = 0
  for i in range(na.shape[0]):
    for j in range(0, na[i]):
      ru[e+j] = sp.symbols("U"+str(i+1)+"."+str(j+1))
    e += na[i]

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

  if root:
    for i in range(nb.shape[0]):
      pprint(sp.sqrt(sp.symbols("Y"+str(i+1)+"."+str(1))))
      M = np.hstack((M, sp.sqrt(sp.symbols("Y"+str(i+1)+"."+str(1)))))
  return M

def matrix_candidate(u, y, nb, na, level, root=False):
    if len(na) != u.shape[0]:
       print("Número de entradas incompativel:", len(na),'e',  u.shape[0])
       return np.array([])
    elif len(nb) != y.shape[0]:
       print("Número de saids incompativel:", len(nb),' e',  y.shape[0])
       return np.array([])

    M = []
    nx = np.sum(na)
    ny = np.sum(nb)
    #print(nx, ny)
    size = nx + ny
    ni = [size]
    t = np.zeros((level, size))
    t[0, :] = np.ones(size)

    for i in range(1, level):
        k = i + 1
        o = ni[i-1] * (nx + ny + k - 1) / k
        ni.append(o)
        s = ni[i-1]
        t[i, 0] = 1
        t[i, 1] = s

        if size > 2:
            for j in range(2, size):
                s = s - t[i-1, j-1]
                t[i, j] = s
    M = []
    H = y.shape[1]#len(y[0])
    #print(H)

    begin = max(max(nb), max(na))

    for i in range(begin, H):
        l = []
        l1 = []
        for j in range(0, y.shape[0]):
          l1 = np.hstack((l1, y[j][i-nb[j]:i]))
        for j in range(u.shape[0]):
          l1 = np.hstack((l1, u[j][i-na[j]:i]))
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
    print(phi.shape, ones.shape)
    phi1 = np.hstack((ones, phi))
    if root:
      for j in range(y.shape[0]):
        aux = y[j, begin:].copy()
        aux[aux < 0] = 0
        #print(aux.shape)
        phi1 = np.hstack((phi1, np.sqrt(aux).reshape((-1,1))))
    return phi1

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

def predict(u, y, theta, model, nb, na, index):
  yest = np.zeros(y.shape)
  d = max(max(na), max(nb))
  yest[:d] = y[:d] #padding

  s = ()
  for i in range(y.shape[0]):
    s += symbols('Y'+str(i+1)+'.1:{}'.format(nb[i]+1))

  for i in range(u.shape[0]):
    s += symbols('U'+str(i+1)+'.1:{}'.format(na[i]+1))

  for k in range(d, y.shape[1]):
    num = np.array([])
    for i in range(y.shape[0]):
      num = np.hstack((num, np.flip(yest[i, k-nb[i]:k])))
    for i in range(u.shape[0]):
      num = np.hstack((num, np.flip(u[i, k-na[i]:k])))
    dicionario = dict(zip(s, num))
    dicionario['1'] = 1
    aux = np.array([m.evalf(subs=dicionario) for m in model])
    yest[index, k] = aux @ theta
  #print(aux.shape, theta.shape)
  return yest[index, :]

#%%
dataTank = pd.read_csv('coupletanks.csv')
print(dataTank.keys())
u = np.reshape(np.array(dataTank['u']), (1,2401))
y = np.array(dataTank[['tank1', 'tank2']].T)

Na = [2]
Nb = [2, 2]
level = 2
output = 0
symCand = symbolic_candidates(Nb, Na, level, True)
phi = matrix_candidate(u, y, Nb, Na, level, True)
print(symCand.shape, phi.shape)
#%%
pad = max(max(Na), max(Nb))
psi, selected = semp(phi, y[output, pad:], 5, 0.00001)

theta = LSM(y[output, pad:], psi)
print("Estrutura selecionada:", symCand[selected], psi.shape)

result = predict(u, y, theta, symCand[selected], Nb, Na, output)

plt.plot(y[output], label='Original')
plt.plot(result, label='Simulação')
plt.legend()
plt.show()

plt.plot(y[output, pad:], label='Original')
plt.plot(psi @ theta, label='Simulação')
plt.legend()
plt.show()
#%%
size = 4
level = 3
r = np.array(sp.symbols("Y:"+str(size)))
pprint(r.tolist()) 

base = []
result = []
aux = np.expand_dims(r, axis=1)
result.append(r)
print(result)

for j in range(level-1):
  base = []
  for i in range(size):
    base.append(np.hstack((aux[i:])))
    #print(np.hstack((aux[i:])))
  aux = []
  for i in range(size):
    aux.append(r[i] * base[i])
  #print(np.hstack((aux)))
  result.append(np.hstack((aux)))
final = np.hstack((result))

if len(set(final.tolist())) == final.shape[0]:
  pprint(final.tolist())
else:
  print("Elemento repetido")
# %%
def symbolic_regressors(nb, na, level, nonlinear=[0,0,0,0]):
  nb = np.array(nb)
  na = np.array(na)
  ny = np.sum(nb)
  nx = np.sum(na)
  
  size = nx + ny

  ry = sp.zeros(1, ny)
  p = 0
  for i in range(nb.shape[0]):
    for j in range(0, nb[i]):
      ry[p+j] = sp.symbols("Y"+str(i+1)+"."+str(j+1))
    p += nb[i]

  yNonlinear = []
  if(nonlinear[0]):
    yNonlinear = yNonlinear + [sp.sin(sp.symbols("Y" + str(s+1) + ".1")) for s in range(nb.shape[0])]
  if(nonlinear[1]):
    yNonlinear = yNonlinear + [sp.cos(sp.symbols("Y" + str(s+1) + ".1")) for s in range(nb.shape[0])]
  if(nonlinear[2]):
    yNonlinear = yNonlinear + [sp.ln(sp.symbols("Y" + str(s+1) + ".1")) for s in range(nb.shape[0])]
  if(nonlinear[3]):
    yNonlinear = yNonlinear + [sp.tan(sp.symbols("Y" + str(s+1) + ".1")) for s in range(nb.shape[0])]#[sp.tan(s) for s in ry]
  

  #print((ry[0:] + yNonlinear), "\n")
  regY = np.array(ry[0:] + yNonlinear)
  #print(regY)
  
  ru = sp.zeros(1, nx)
  e = 0
  for i in range(na.shape[0]):
    for j in range(0, na[i]):
      ru[e+j] = sp.symbols("U"+str(i+1)+"."+str(j+1))
    e += na[i]

  uNonlinear = []
  if(nonlinear[0]):
    uNonlinear = uNonlinear + [sp.sin(sp.symbols("U" + str(s+1) + ".1")) for s in range(na.shape[0])]
  if(nonlinear[1]):
    uNonlinear = uNonlinear + [sp.cos(sp.symbols("U" + str(s+1) + ".1")) for s in range(na.shape[0])]
  if(nonlinear[2]):
    uNonlinear = uNonlinear + [sp.ln(sp.symbols("U" + str(s+1) + ".1")) for s in range(na.shape[0])]
  if(nonlinear[3]):
    uNonlinear = uNonlinear + [sp.tan(sp.symbols("U" + str(s+1) + ".1")) for s in range(na.shape[0])]#[sp.tan(s) for s in ry]
  regU = np.array(ru[0:] + uNonlinear)
  #print(regU)
  l1 = np.hstack((regY, regU))
  #print(l1)

  '''l1 = np.array(np.hstack((ry, ru))[0])
  print(l1)'''

  base = []
  result = []
  aux = np.expand_dims(l1, axis=1)
  result.append(l1)
  print(size, l1.shape)
  num = l1.shape[0]

  for j in range(level-1):
    base = []
    for i in range(num):
      base.append(np.hstack((aux[i:])))
    aux = []
    for i in range(num):
      aux.append(l1[i] * base[i])
    result.append(np.hstack((aux)))
  final = np.hstack((result))
  '''
  if len(set(final.tolist())) == final.shape[0]:
    pprint(final.tolist())
  else:
    print("Elemento repetido")'''
  final = np.hstack((1, final))
  return final

pprint(symbolic_regressors([1,1], [1], 2, [0,0,1,0]))
#pprint(symbolic_candidates([1, 1], [1], 2))
#%%

def matrix_candidate2(u, y, nb, na, level, nonlinear=[0,0,0,0,0,0]):
    if len(na) != u.shape[0]:
       print("Número de entradas incompativel:", len(na),'e',  u.shape[0])
       return np.array([])
    elif len(nb) != y.shape[0]:
       print("Número de saids incompativel:", len(nb),' e',  y.shape[0])
       return np.array([])
    
    def exp(x):
      return np.exp(x/8)
    def squareRootM(x):
      if x <= 0:
        return 0
      else:
        return np.sqrt(x)
    functions = [np.sin, np.cos, np.log, np.tanh, exp, squareRootM]
    M = []
    nx = np.sum(na)
    ny = np.sum(nb)
    size = nx + ny + len(Nb) * np.sum(nonlinear) + len(Na) * np.sum(nonlinear)
    ni = [size]

    M = []
    H = y.shape[1]#len(y[0])
    #print("Tamanho da entrada:", H)

    begin = max(max(nb), max(na))
    
    regY = np.zeros((ny, H - begin))
    for i in range(len(Nb)):
      for j in range(1, Nb[i] + 1):
        regY[i*2 + j - 1] = y[i][begin-j:-j]

    for j in range(len(nonlinear)):
      if nonlinear[j]:
        for i in range(len(Nb)):
          #print(functions[j](y[i][begin-1:-1]))
          regY = np.vstack((regY, functions[j](y[i][begin-1:-1])))

    regU = np.zeros((nx, H - begin))
    for i in range(len(Na)):
      for j in range(1, Na[i] + 1):
        regU[i*2 + j - 1] = u[i][begin-j:-j]

    for j in range(len(nonlinear)):
      if nonlinear[j]:
        for i in range(len(Na)):
          regU = np.vstack((regU, functions[j](u[i][begin-1:-1])))
    
    print(regU.shape, nx)
    print(regY.shape, ny)

    l1 = np.vstack((regY, regU))
    result = []
    aux = np.expand_dims(l1, axis=1)
    result = l1.copy()
    print(size, ny+nx, l1.shape, aux.shape)
    num = l1.shape[0]

    
    for j in range(level-1):
      base = []
      for i in range(num):
        base.append(np.vstack((aux[i:])))
      aux = []
      #print(len(base), base[0].shape, l1[0].shape)
      for i in range(num):
        #print(l1[i].shape, base[i].shape, (l1[i] * base[i]).shape)
        aux.append(l1[i] * base[i])
      #print("aux: ", len(aux), aux[0].shape, len(result), np.vstack((aux)).shape)
      inn = np.vstack((aux))
      result = np.vstack((result, inn))
    final = np.vstack((result))
    ones = np.ones((1, l1.shape[1]))
    final = np.vstack((ones, final))
    print(final.shape)
    return final
   
non = [1,1,0,0,0,0]
sCandidatos = symbolic_regressors(Nb, Na, level, non)
print(sCandidatos)
vCandidatos = matrix_candidate2(u, y, Nb, Na, level, non)
#%%
for i in range(sCandidatos.shape[0]):
  print(i, sCandidatos[i])

index = [9,8,8]
print(np.prod(vCandidatos[217] == np.prod(vCandidatos[index], axis = 0)))
# %%
output = 0

pad = max(max(Na), max(Nb))
Psi, selected = semp(vCandidatos.T, y[output, pad:], 5, 0.00001)
theta = LSM(y[output, pad:], Psi)
print(sCandidatos[selected], theta)
# %%
plt.plot(Psi @ theta)
plt.plot(y[output, pad:])
plt.plot(y[output, pad:] - Psi @ theta) 
plt.show()
#%%
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, pprint
import matplotlib.pyplot as plt
from methods.utils.utilities import *

dataTank = pd.read_csv('coupletanks.csv')
u = np.reshape(np.array(dataTank['u']), (1,2401))
y = np.array(dataTank[['tank1', 'tank2']].T)

Na = [2]
Nb = [1,1]#[2, 2]
level = 2
output = 0

from structureSelector import *
sselector = structureSelector()
sselector.symbolic_regressors(Nb, Na, level,[0,0,0,0,0], True)
# %%

def matrix_candidate2(u, y, nb, na, level, nonlinear=[0,0,0,0,0], root=False):
	if len(na) != u.shape[0]:
	   print("Número de entradas incompativel:", len(na),'e',  u.shape[0])
	   return np.array([])
	elif len(nb) != y.shape[0]:
	   print("Número de saids incompativel:", len(nb),' e',  y.shape[0])
	   return np.array([])
	
	def exp(x):
	  return np.exp(x/8)
	def squareRootM(x):
	  if x <= 0:
		  return 0
	  else:
		  return np.sqrt(x)
	functions = [np.sin, np.cos, np.log, np.tanh, exp, squareRootM]
	
	M = []
	nx = np.sum(na)
	ny = np.sum(nb)
	size = nx + ny + len(Nb) * np.sum(nonlinear) + len(Na) * np.sum(nonlinear)
	ni = [size]

	M = []
	H = y.shape[1]#len(y[0])

	begin = max(max(nb), max(na))
	
	regY = np.zeros((ny, H - begin))
	k = 0
	for i in range(len(Nb)):
	  for j in range(1, Nb[i] + 1):
		  #print(k, i*2 + j - 1)
		  regY[k] = y[i][begin-j:-j]
		  k += 1

	for j in range(len(nonlinear)):
	  if nonlinear[j]:
		  for i in range(len(Nb)):
		    regY = np.vstack((regY, functions[j](y[i][begin-1:-1])))

	regU = np.zeros((nx, H - begin))
	k = 0
	for i in range(len(Na)):
	  for j in range(1, Na[i] + 1):
		  #regU[i*2 + j - 1] = u[i][begin-j:-j]
		  regU[k] = u[i][begin-j:-j]
		  k += 1

	for j in range(len(nonlinear)):
	  if nonlinear[j]:
		  for i in range(len(Na)):
			    regU = np.vstack((regU, functions[j](u[i][begin-1:-1])))

	l1 = np.vstack((regY, regU))
	result = []
	aux = np.expand_dims(l1, axis=1)
	result = l1.copy()
	num = l1.shape[0]
	
	for j in range(level-1):
	  base = []
	  for i in range(num):
		  base.append(np.vstack((aux[i:])))
	  aux = []
	  for i in range(num):
		  aux.append(l1[i] * base[i])
	  inn = np.vstack((aux))
	  result = np.vstack((result, inn))
	final = np.vstack((result))
	ones = np.ones((1, l1.shape[1]))
	final = np.vstack((ones, final))
	if root:
		yy = y[:, begin-1:-1].copy()
		yy[yy < 0] = 0
		uu = u[:, begin-1:-1].copy()
		uu[uu < 0] = 0
		x = np.vstack((yy,uu))
		r = np.sqrt(x)
		#print(yy.shape, uu.shape, x.shape)
		'''plt.plot(y.T)
		plt.plot(r.T)
		plt.show()'''
		final = np.vstack((final, r))
	return final

vCandidatos = matrix_candidate2(u, y, Nb, Na, level, fn, root=True)
print(vCandidatos.shape)