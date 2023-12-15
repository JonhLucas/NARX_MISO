#%%
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, pprint, Function
from methods.utils.utilities import *

class sqrtM(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			#print(type(x))
			return sp.re(sp.sqrt(x))
           
class structureSelector:

	def exp(self, x):
		return sp.exp(x/8)
		
	def logM(self, x):
		if x < 0:
			return -np.inf
		else:
			return sp.log(x)

	functions = [sp.sin, sp.cos, sp.log, sp.tanh, exp]

	def symbolic_regressors(self, nb, na, level, nonlinear=[0,0,0,0,0], root=False, delay=0):
		nb = np.array(nb)
		na = np.array(na)
		ny = np.sum(nb)
		nx = np.sum(na - delay)
		
		#Regressores da saída
		ry = sp.zeros(1, ny)
		p = 0
		for i in range(nb.shape[0]):
			for j in range(0, nb[i]):
				ry[p+j] = sp.symbols("Y"+str(i+1)+"."+str(j+1))
			p += nb[i]

		yNonlinear = []
		#Funções não linear aplicadas a y
		for i in range(len(nonlinear)):
			if(nonlinear[i]):
				yNonlinear = yNonlinear + [self.functions[i](sp.symbols("Y" + str(s+1) + ".1")) for s in range(nb.shape[0])]
		#junção
		regY = np.array(ry[0:] + yNonlinear)
		
		#Regressores de entrada
		ru = sp.zeros(1, nx)
		e = 0
		for i in range(na.shape[0]):
			for j in range(0, na[i]-delay):
				ru[e+j] = sp.symbols("U"+str(i+1)+"."+str(j+1+delay))
			e += (na[i] - delay)
		#Funções não lineares aplicadas a entrada
		uNonlinear = []
		for i in range(len(nonlinear)):
			if(nonlinear[i]):
				uNonlinear = uNonlinear + [self.functions[i](sp.symbols("U" + str(s+1) + ".1")) for s in range(na.shape[0])]
		#junção
		regU = np.array(ru[0:] + uNonlinear)
		
		l1 = np.hstack((regY, regU))
	
		base = []
		result = []
		aux = np.expand_dims(l1, axis=1)
		result.append(l1)
		#print(size, l1.shape)
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
		final = np.hstack((1, final))
	
		if root:
			r = []
			r = r + [sqrtM(sp.symbols("Y" + str(s+1) + ".1")) for s in range(nb.shape[0])] + [sqrtM(sp.symbols("U" + str(s+1) + ".1")) for s in range(na.shape[0])]
			
			final = np.hstack((final, r))
			print(r, final)
	
		return final
	
	def matrix_candidate(self, u, y, nb, na, level, nonlinear=[0,0,0,0,0], root=False, delay=0):
		if len(na) != u.shape[0]:
			print("Número de entradas incompativel:", len(na),'e',	u.shape[0])
			return np.array([])
		elif len(nb) != y.shape[0]:
			print("Número de saids incompativel:", len(nb),' e',	y.shape[0])
			return np.array([])
		
		def exp(x):
			return np.exp(x/8)

		#apagar
		functions = [np.sin, np.cos, np.log, np.tanh, exp]
	
		M = []
		nx = np.sum(np.array(na) - delay)
		ny = np.sum(nb)
		size = nx + ny + len(nb) * np.sum(nonlinear) + len(na) * np.sum(nonlinear)

		H = y.shape[1]#len(y[0])
	
		begin = max(max(nb), max(na))
		
		regY = np.zeros((ny, H - begin))
		k = 0
		for i in range(len(nb)):
			for j in range(1, nb[i] + 1):
				#print(k, i*2 + j - 1)
				regY[k] = y[i][begin-j:-j]
				k += 1
	
		for j in range(len(nonlinear)):
			if nonlinear[j]:
				for i in range(len(nb)):
					regY = np.vstack((regY, functions[j](y[i][begin-1:-1])))
	
		regU = np.zeros((nx, H - begin))
		k = 0
		for i in range(len(na)):
			for j in range(1+delay, na[i] + 1):
				#regU[i*2 + j - 1] = u[i][begin-j:-j]
				#print(j)
				regU[k] = u[i][begin-j:-j]
				k += 1
		#print("*************Size:", regU.shape)
		#print(regU)
		for j in range(len(nonlinear)):
			if nonlinear[j]:
				for i in range(len(na)):
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
			x[x < 0] = 0
			r = np.sqrt(x)
			final = np.vstack((final, r))
		return final
	
	def semp(self, psi, y, ni, rho = 0.00001):
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
				theta = (np.linalg.inv(p.T @ p) @ p.T) @ y
				J = np.append(J, np.mean(np.square(y - (p @ theta))))
			l = np.argmin(J)
			if J[l] < Jold and np.abs(J[l] - Jold) > rho:
				if P.shape[0] == 0:
					P = np.append(P, Q[:, l]).reshape((-1,1))
				else:
					P = np.hstack((P, Q[:, l].reshape((-1,1))))
				Q = np.delete(Q, l, 1)
				selected.append(idx[l])
				idx = np.delete(idx, l)
			else:
				return P, selected
	
			flag = True
			while P.shape[1] > 1 and flag:
				Jp = np.array([])
				for k in range(P.shape[1]):
					R = np.delete(P, k, 1)
					theta = (np.linalg.inv(R.T @ R) @ R.T) @ y
					Jp = np.append(Jp, np.mean(np.square(y - (R @ theta))))
				m = np.argmin(Jp)
				if Jp[m] < Jold:
					P = np.delete(P, m, 1)
					selected.pop(m)
					continue
				else:
					flag = False #revisar
			#atualizando Jold
			theta = (np.linalg.inv(P.T @ P) @ P.T) @ y
			Jold = np.mean(np.square(y - (P @ theta)))#J[l]
		return P, selected

	def predict(self, u, y, theta, model, nb, na, index):
		#Condição inicial
		yest = np.zeros(y.shape)
		d = max(max(na), max(nb))
		yest = y.copy()
		yest[index, :] = 0
		yest[index, :d] = y[index, :d] #padding

		#
		nb = np.array(nb)
		nb[nb == 0] = 1
		na = np.array(na)
		na[na == 0] = 1

		'''plt.plot(yest.T, label=["1","2"])
		plt.legend()
		plt.show()'''

		s = []
		for i in range(nb.shape[0]):
			for j in range(nb[i]):
				s += [symbols('Y'+str(i+1)+'.'+str(j+1))]
	
		for i in range(u.shape[0]):
			s += symbols('U'+str(i+1)+'.1:{}'.format(na[i]+1))
		
		#print('--------s: ', s)
		for k in range(d, y.shape[1]):
			num = np.array([])
			for i in range(y.shape[0]):
				num = np.hstack((num, np.flip(yest[i, k-nb[i]:k])))
			for i in range(u.shape[0]):
				num = np.hstack((num, np.flip(u[i, k-na[i]:k])))
			dicionario = dict(zip(s, num))
			aux = np.array([1 if m == 1 else m.evalf(subs=dicionario) for m in model])
			#print(aux)
			#print(np.real(aux[-1]), type(aux[-1]))
			yest[index, k] = aux.real @ theta
		return yest[index, :]
	
	def oneStepForward(self, u, y, theta, model, nb, na, index):
		#Condição inicial
		yest = np.zeros(y.shape)
		d = max(max(na), max(nb))
		yest = y.copy()
		yest[index, :] = 0
		yest[index, :d] = y[index, :d] #padding

		s = []
		nb = np.array(nb)
		nb[nb == 0] = 1
		na = np.array(na)
		na[na == 0] = 1
		
		for i in range(nb.shape[0]):
			for j in range(nb[i]):
				s += [symbols('Y'+str(i+1)+'.'+str(j+1))]
	
		for i in range(u.shape[0]):
			s += symbols('U'+str(i+1)+'.1:{}'.format(na[i]+1))
		
		#print('--------s: ', s)
		for k in range(d, y.shape[1]):
			num = np.array([])
			for i in range(y.shape[0]):
				num = np.hstack((num, np.flip(y[i, k-nb[i]:k])))
			for i in range(u.shape[0]):
				num = np.hstack((num, np.flip(u[i, k-na[i]:k])))
			dicionario = dict(zip(s, num))
			aux = np.array([1 if m == 1 else m.evalf(subs=dicionario) for m in model])
			yest[index, k] = aux @ theta
		return yest[index, :]

#%%
'''
na = [5]
nb = [2,2]
level = 2
ss = structureSelector()
d = 3
s = ss.symbolic_regressors(nb, na, level, nonlinear=[0,0,0,0,0], root=True, delay=d)
pprint(s)
print(s[-1].evalf(subs={symbols('U1.1'):-3}))
u = np.arange(1,1001,1).reshape((1,-1))
y = np.zeros((2,1000))
v = ss.matrix_candidate(u, y, nb, na, level, delay=d)
print(s.shape, v.shape)'''
# %%
