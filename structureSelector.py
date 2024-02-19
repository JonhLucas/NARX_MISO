#%%
import numpy as np
import pandas as pd
import sympy as sp
from sympy import symbols, pprint, Function, Derivative
from methods.utils.utilities import *

class sqrtM(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			#print(type(x))
			return sp.re(sp.sqrt(x))
		
class sign(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			#print(type(x))
			return np.sign(x)

class d(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			#print(type(x))
			print(x)
			return x


class structureSelector:
	def integrate(self, p, ts=0.1):
		r = np.zeros(p.shape)
		r[:, 0] = p[:, 0] * ts
		#print(r[:, 0], p[:, 0])
		for i in range(1, p.shape[1]):
			r[:, i] = r[:, i-1] + p[:, i] * ts
		return r

	def exp(self, x):
		return sp.exp(x/8)

	functions = [sp.sin, sp.cos, sp.log, sp.tanh, sign]

	def regressors(self, size, n, symbol="", nl=[0,0,0,0,0], df=False, d=0, intg=False):
		r = sp.zeros(1, size)
		p = 0
		#Regressores lineares
		for i in range(n.shape[0]):
			for j in range(0, n[i]-d):
				r[p+j] = sp.symbols(symbol+str(i+1)+"."+str(j+1+d))
			p += n[i]-d

		#Aplicação das funções não lineares
		sNonlinear = []
		for i in range(len(nl)):
			if(nl[i]):
				sNonlinear = sNonlinear + [self.functions[i](sp.symbols(symbol + str(s+1) + ".1")) for s in range(n.shape[0])]
		
		#Aplicação da derivada
		ds = []
		if df:
			#f = Function('d')
			#ds = [f(sp.symbols(symbol + str(s+1) + ".1")) for s in range(n.shape[0])]
			ds = [sp.symbols("d(" + symbol + str(s+1) + ".1)") for s in range(n.shape[0])]

		#Aplicação da integral
		Is = []
		if intg:
			Is = [sp.symbols("I(" + symbol + str(s+1) + ".1)") for s in range(n.shape[0])]
		#print("integral", Is)

		#Junção
		regS = np.array(r[0:] + sNonlinear + ds + Is) #np.array(ry[0:] + yNonlinear)
		#print('Regressores gerados:', regS)
		return regS

	def symbolic_regressors(self, nb, na, level, nonlinear=[0,0,0,0,0], root=False, delay=0, diff=False, intg=False):
		nb = np.array(nb)
		na = np.array(na)
		ny = np.sum(nb)
		nx = np.sum(na - delay)
		
		#regressores de saída
		regY = self.regressors(ny, nb, "Y", nonlinear, diff, 0, intg)
		#regressores de entrada
		regU = self.regressors(nx, na, "U", nonlinear, diff, delay, intg)
		#regressores lineares
		l1 = np.hstack((regY, regU))
	
		#Agrupamento em termos polinomiais
		base = []
		result = []
		aux = np.expand_dims(l1, axis=1)
		result.append(l1)
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
			#print(r, final)
	
		return final
	
	def matrix_candidate(self, u, y, nb, na, level, nonlinear=[0,0,0,0,0], root=False, delay=0, diff=False, dt=0, intg=False):
		#Verificação inicial
		if len(na) != u.shape[0]:
			print("Número de entradas incompativel: S-", len(na),'e U-',	u.shape[0])
			return np.array([])
		elif len(nb) != y.shape[0]:
			print("Número de saids incompativel:", len(nb),' e',	y.shape[0])
			return np.array([])

		def exp(x):
			return np.exp(x/8)

		#apagar
		functions = [np.sin, np.cos, np.log, np.tanh, np.sign]
	
		#M = []
		nx = np.sum(np.array(na) - delay)
		ny = np.sum(nb)
		#size = nx + ny + len(nb) * np.sum(nonlinear) + len(na) * np.sum(nonlinear)

		H = y.shape[1]#len(y[0])
	
		begin = max(max(nb), max(na))
		#Regressores lineares
		#regresoores de saída 
		regY = np.zeros((ny, H - begin))
		k = 0
		for i in range(len(nb)):
			for j in range(1, nb[i] + 1):
				regY[k] = y[i][begin-j:-j]
				k += 1	

		#não lineares
		for j in range(len(nonlinear)):
			if nonlinear[j]:
				for i in range(len(nb)):
					regY = np.vstack((regY, functions[j](y[i][begin-1:-1])))
		#diferencial
		if diff:
			if dt == 0:
				print("Erro encontrado. Informe o intervalo de amostragem.")
				return []
			else:
				dy = np.zeros((len(nb), H - begin))
				#print(ny, dy.shape, y.shape,y[:, begin-1:-1].shape)
				dy = (y[:, begin-1:-1]-y[:, begin-2:-2])/dt
				
				regY = np.vstack((regY, dy))
		#integracao da saida
		if intg:
			iy = self.integrate(p=y, ts=dt)
			'''
			plt.plot(iy[:].T )
			plt.show()
			print(iy[:, begin:].shape, regY.shape)'''
			regY = np.vstack((regY, iy[:, begin:]))

		#regressores de entrada
		regU = np.zeros((nx, H - begin))
		k = 0
		for i in range(len(na)):
			for j in range(1+delay, na[i] + 1):
				regU[k] = u[i][begin-j:-j]
				k += 1

		for j in range(len(nonlinear)):
			if nonlinear[j]:
				for i in range(len(na)):
						regU = np.vstack((regU, functions[j](u[i][begin-1:-1])))
		#diferencial
		if diff:
			if dt == 0:
				print("Erro encontrado. Informe o intervalo de amostragem.")
				return []
			else:
				du = np.zeros((len(na), H - begin))
				#print(nx, du.shape, u.shape, u[:, begin-1:-1].shape)
				du = (u[:, begin-1:-1]-u[:, begin-2:-2])/dt
				regU = np.vstack((regU, du))
		#integracao da entrada
		if intg:
			iu = self.integrate(p=u, ts=dt)
			regU = np.vstack((regU, iu[:, begin:]))

		l1 = np.vstack((regY, regU))
		result = []
		aux = np.expand_dims(l1, axis=1)
		result = l1.copy()
		num = l1.shape[0]
		#merge
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

	def predict(self, u, y, theta, model, nb, na, index, delay=0, diff=False, dt=0, intg=False):
		#Condição inicial
		print("Simulação livre")
		d = max(max(na), max(nb))
		d = max(d, delay)

		yest = np.zeros(y.shape)
		#print('condições iniciais', yest[index, :d], y[index, :d])
		yest[index, :d] = y[index, :d] #padding
		if y.shape[0] > 1:
			w = np.arange(0, y.shape[0], 1)
			w = np.delete(w, index)
			print(index, w)
			yest[w, :] = y[w, :] 

		#
		nb = np.array(nb)
		nb[nb == 0] = 1
		na = np.array(na)
		na[na == 0] = 1


		s = []
		for i in range(nb.shape[0]):
			for j in range(nb[i]):
				s += [symbols('Y'+str(i+1)+'.'+str(j+1))]
		if diff:
			for i in range(nb.shape[0]):
				s += [symbols('d(Y'+str(i+1)+'.'+'1)')]
		if intg:
			for i in range(nb.shape[0]):
				s += [symbols('I(Y'+str(i+1)+'.'+'1)')]


		for i in range(u.shape[0]):
			s += symbols('U'+str(i+1)+'.'+str(delay)+':{}'.format(na[i]+1))

		if diff:
			for i in range(na.shape[0]):
				s += [symbols('d(U'+str(i+1)+'.'+'1)')]
			du = np.zeros(u.shape)
			du[:, d:] = (u[:, d-1:-1] - u[:, d-2:-2]) / dt

		if intg:
			for i in range(na.shape[0]):
				s += [symbols('I(U'+str(i+1)+'.'+'1)')]
			iu = np.zeros(u.shape)
			iu[:, 0] = u[:, 0] * dt
			for i in range(1, d):
				iu[:, i] = iu[:, i-1] + u[:, i] * dt
		
		print('--------s: ', s)
		iy = 0
		for k in range(d, y.shape[1]):
			num = np.array([])
			for i in range(y.shape[0]):
				num = np.hstack((num, np.flip(yest[i, k-nb[i]:k])))
			if diff:
				dy = (yest[:, k-1] - yest[:, k-2]) / dt
				#print(num.shape, dy.shape)
				num = np.hstack((num, dy))
			if intg:
				iy = iy + yest[:, k-1] * dt
				num = np.hstack((num, iy))

			for i in range(u.shape[0]):
				num = np.hstack((num, np.flip(u[i, k-na[i]:k-delay+1])))
			if diff:
				num = np.hstack((num, du[:, k]))
			if intg:
				iu[:, k] = iu[:, k-1] + u[:, k] * dt
				num = np.hstack((num, iu[:, k]))

			dicionario = dict(zip(s, num))
			#print(dicionario)
			#print(len(s), num.shape, k,  k-na[i], k-delay+1)
			aux = np.array([1 if m == 1 else m.evalf(subs=dicionario) for m in model])
			#print(aux)
			#print(np.real(aux[-1]), type(aux[-1]))
			try:
				yest[index, k] = aux.real @ theta
			except ValueError:
				print("Error:")
				print(aux)
			except Exception as err:
				print(f"Unexpected {err=}, {type(err)=}")
				print(aux)
				
		return yest[index, :]
	
	def oneStepForward(self, u, y, theta, model, nb, na, index, diff=False, dt=0, intg=False):
		#Condição inicial
		#print("oneStepForward")
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
		if diff:
			for i in range(nb.shape[0]):
				s += [symbols('d(Y'+str(i+1)+'.'+'1)')]
		if intg:
			for i in range(nb.shape[0]):
				s += [symbols('I(Y'+str(i+1)+'.'+'1)')]
			iy = self.integrate(y, dt)

		for i in range(u.shape[0]):
			s += symbols('U'+str(i+1)+'.1:{}'.format(na[i]+1))

		if diff:
			for i in range(na.shape[0]):
				s += [symbols('d(U'+str(i+1)+'.'+'1)')]
		
			dy = np.zeros(y.shape)
			dy[:, d:] = (y[:, d-1:-1] - y[:, d-2:-2]) / dt
			du = np.zeros(u.shape)
			du[:, d:] = (u[:, d-1:-1] - u[:, d-2:-2]) / dt

		if intg:
			for i in range(na.shape[0]):
				s += [symbols('I(U'+str(i+1)+'.'+'1)')]
			iu = self.integrate(u, dt)
		
		
		#print('--------s: ', s, dy.shape)
		for k in range(d, y.shape[1]):
			num = np.array([])
			for i in range(y.shape[0]):
				num = np.hstack((num, np.flip(y[i, k-nb[i]:k])))
			if diff:
				num = np.hstack((num, dy[:, k]))
			if intg:
				num = np.hstack((num, iy[:, k]))
			for i in range(u.shape[0]):
				num = np.hstack((num, np.flip(u[i, k-na[i]:k])))
			if diff:
				num = np.hstack((num, du[:, k]))
			if intg:
				num = np.hstack((num, iu[:, k]))

			dicionario = dict(zip(s, num))
			#print(dicionario)
			aux = np.array([1 if m == 1 else m.evalf(subs=dicionario) for m in model])
			#print(aux, dicionario, num.shape, len(s), nb, na)
			yest[index, k] = aux @ theta
		return yest[index, :]

#%%
'''
na = [2]
nb = [2,2]
level = 2
ss = structureSelector()
d = 1
s = ss.symbolic_regressors(nb, na, level, nonlinear=[0,0,0,0,0], root=False, diff=False, delay=d, intg=True)
pprint(s)
print(len(s))
#%%
na = [2]
nb = [2,2]
ss = structureSelector()
#print(s[-1].evalf(subs={symbols('U1.1'):-3}))
u = np.arange(0,1000,1).reshape((1,-1))/10
y = np.zeros((2,1000))
y[0] = np.sin(u)
y[1] = 2*u
dl = 1
v = ss.matrix_candidate(u, y, nb, na, level, root=False, diff=False, delay=dl, intg=True, dt=0.1)
print(len(s), v.shape)
#%%
plt.plot(np.sin(u).T)
plt.plot(-np.cos(u).T)
plt.show()
#%%
output = 0
pad = max(max(na), max(nb))
psi, selected  = ss.semp(v.T, y[output, pad:], 3, 0.00001)
theta = LSM(y[output, pad:], psi)
model = s[selected]
print(model, theta)
#%%
theta = []
model = []
index = 0
t = ss.predict(u, y, theta, model, nb, na, index, diff=True, dt=0.1)
#%%
plt.plot(v[:,:100].T)
plt.show()'''
