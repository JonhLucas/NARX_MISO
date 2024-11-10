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
			return sp.re(sp.sqrt(x))
		
class sign(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			return np.sign(x)

class d(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			return x

class tanh05(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			return sp.tanh(0.5 * x)
			
class tanh2(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			return sp.tanh(2 * x)
		
class tanh5(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			return sp.tanh(5 * x)
		
class tanh10(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			return sp.tanh(10 * x)

class clip(Function):
	m = 0 
	n = 5 

	@classmethod
	def setLimit(cls, min_val, max_val):
		cls.m = min_val
		cls.n = max_val

	@classmethod
	def eval(cls, x):
		#print('----------------clip--------------', type(x))
		if isinstance(x, np.ndarray):
			return np.clip(x, cls.m, cls.n)
		elif x.is_Number:
			return np.clip(x, cls.m, cls.n)
	def __str__(self):
		return f"c({self.args[0]})"
		
class sen(Function):
	@classmethod
	def eval(cls, x):
		if isinstance(x, np.ndarray):
			return np.sin(x)
		elif x.is_Number:
			return np.sin(x)
		
	def __str__(self):
		return f"sin({self.args[0]})"

class fourthRoot(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			if x < 0:
				x = 0
			return sp.root(x, 4)
		return x**(sp.Rational(1, 4))
	
class squareRoot(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			if x < 0:
				x = 0
			#print(x)
			return sp.root(x, 2)
		return sp.re(x**(sp.Rational(1, 2)))
	
class sqrt(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			if x < 0:
				x = 0
			#print(x)
			return sp.re(sp.root(x, 2))
	
class abs(Function):
	@classmethod
	def eval(cls, x):
		if x.is_Number:
			return np.abs(x)

class structureSelector:
	def __init__(self):
		self.min = -1
		self.max = 1

	functions = [sp.sin, sp.cos, sp.log, sp.tanh, sign, tanh2, tanh5, tanh10, tanh05, sqrt, abs]
	
	def setLimits(self, min, max):
		self.min = min
		self.max = max
		clip.setLimit(min, max)

	def integrate(self, p, ts=0.1):
		r = np.zeros(p.shape)
		r[:, 0] = p[:, 0] * ts
		for i in range(1, p.shape[1]):
			r[:, i] = r[:, i-1] + p[:, i] * ts
		return r

	def exp(self, x):
		return sp.exp(x/8)

	def regressors(self, size, n, symbol="", nl=[0,0,0,0,0,0,0,0,0, 0], d=0, ymodifier=[0,0]):
		r = sp.zeros(1, size)
		p = 0
		modifier = [clip, sp.sqrt, abs]
		#Regressores lineares
		for i in range(n.shape[0]):
			if n[i]:
				for j in range(0, n[i]-d):
					aux = sp.symbols(symbol+str(i+1)+"."+str(j+1+d))

					for k in range(len(ymodifier)):
						if ymodifier[k]:
							aux = modifier[k](aux)

					r[p+j] = aux
				p += n[i]-d

		#Aplicação das funções não lineares
		sNonlinear = []
		
		for i in range(len(nl)):
			sy = []
			for lk in range(n.shape[0]):	#numero de saidas
				sy += [sp.symbols(symbol + str(lk+1) + "." + str(s+1)) for s in range(nl[i])]
			for k in range(len(ymodifier)):
				if ymodifier[k]:
					sy = [modifier[k](s) for s in sy]
			sy = [self.functions[i](s) for s in sy]
			
			sNonlinear = sNonlinear + sy

		#Junção
		regS = np.array(r[0:] + sNonlinear) #np.array(ry[0:] + yNonlinear)
		#print('Regressores gerados:', regS)
		return regS

	def symbolic_regressors(self, nb, na, level, ynonlinear=[0,0,0,0,0,0,0,0,0,0], unonlinear=[0,0,0,0,0,0,0,0,0,0], root=False, delay=0, ymodifier=[0,0]):
		nb = np.array(nb)
		na = np.array(na)
		ny = np.sum(nb)
		nx = np.sum(na - delay)
		
		#regressores de saída
		regY = self.regressors(ny, nb, "Y", ynonlinear, 0, ymodifier)
		#print(regY)
		#regressores de entrada
		nn = unonlinear.copy()
		nn[-1] = 0
		regU = self.regressors(nx, na, "U", nn, delay)
		#print(regU)
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
			for i in range(nb.shape[0]):
				if nb[i] or True:
					r = r + [sqrtM(sp.symbols("Y" + str(i+1) + ".1"))]

			for i in range(na.shape[0]):
				if na[i]:
					r = r + [sqrtM(sp.symbols("U" + str(i+1) + ".1"))]

			#r = r + [sqrtM(sp.symbols("Y" + str(s+1) + ".1")) for s in range(nb.shape[0])] + [sqrtM(sp.symbols("U" + str(s+1) + ".1")) for s in range(na.shape[0])]
			final = np.hstack((final, r))
			#print(r, final)
	
		return final
	
	def matrix_candidate(self, u, y, nb, na, level, ynonlinear=[0,0,0,0,0,0,0,0,0,0], unonlinear=[0,0,0,0,0,0,0,0,0,0], root=False, delay=0, dt=0, ymodifier=[0,0]):
		#Verificação inicial
		if len(na) != u.shape[0]:
			print("Número de entradas incompativel: S-", len(na),'e U-',	u.shape[0])
			return np.array([])
		elif len(nb) != y.shape[0]:
			print("Número de saids incompativel:", len(nb),' e',	y.shape[0])
			return np.array([])

		def tanh05Numeric(x):
			return np.tanh(0.5 * x)
		
		def tanh2Numeric(x):
			return np.tanh(2 * x)
		
		def tanh5Numeric(x):
			return np.tanh(5 * x)
		
		def tanh10Numeric(x):
			return np.tanh(10 * x)
		
		def clipNumeric(x):
			return np.clip(x, self.min, self.max)
		
		def sqrtNumeric(x):
			x[x < 0] = 0
			return np.sqrt(x)	

		def fourthRoot(x):
			x[x < 0] = 0
			return x**(1/4)
		
		modifier = [clipNumeric, sqrtNumeric, np.abs]
		#conjunto de não linearidades
		functions = [np.sin, np.cos, np.log, np.tanh, np.sign, tanh2Numeric, tanh5Numeric, tanh10Numeric, tanh05Numeric, sqrtNumeric, np.abs]
	
		nx = np.sum(np.array(na) - delay)
		ny = np.sum(nb)

		H = y.shape[1]
	
		begin = max(max(nb), max(na))
		begin = max(begin, max(unonlinear))
		begin = max(begin, max(ynonlinear))

		#Regressores lineares
		#print(nonlinear)
		#regresoores de saída 
		regY = np.zeros((ny, H - begin))
		k = 0
		for i in range(len(nb)):
			for j in range(1, nb[i] + 1):
				regY[k] = y[i][begin-j:-j]
				k += 1	

		for i in range(len(ymodifier)):
			if ymodifier[i]:
				regY = modifier[i](regY)

		#não lineares
		for j in range(len(ynonlinear)):
			for i in range(len(nb)):
				#if nb[i]:
				for k in range(0, ynonlinear[j]):
					arg = y[i][begin-(1 + k):-(1 + k)]
					for l in range(len(ymodifier)):
						if ymodifier[l]:
							arg = modifier[l](arg)
					regY = np.vstack((regY, functions[j](arg)))

		#regressores de entrada
		
		regU = np.zeros((nx, H - begin))
		k = 0
		for i in range(len(na)):
			for j in range(1+delay, na[i] + 1):
				regU[k] = u[i][begin-j:-j]
				k += 1
		
		nn = unonlinear.copy()
		nn[-1] = 0
		for j in range(len(nn)):
			for i in range(len(na)):
				for k in range(0, nn[j]):
						regU = np.vstack((regU, functions[j](u[i][begin-(1+k):-(1+k)])))

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
			yy = y[np.array(nb) > 0, begin-1:-1].copy()
			yy[yy < 0] = 0

			uu = u[np.array(na) > 0, begin-1:-1].copy()
			uu[uu < 0] = 0

			x = np.vstack((yy,uu))
			x[x < 0] = 0
			r = np.sqrt(x)
			final = np.vstack((final, r))
		return final
	
	def generate_candidate(self, u, y, nb, na, level, ynonlinear=[0,0,0,0,0,0,0,0,0,0], unonlinear=[0,0,0,0,0,0,0,0,0,0], root=False, delay=0, ymodifier=[0,0]):
		symbolic = self.symbolic_regressors(nb, na, level, ynonlinear, unonlinear, root, delay, ymodifier)
		mc = self.matrix_candidate(u, y, nb, na, level, ynonlinear, unonlinear, root, delay, ymodifier)

		droped = []
		for i in range(len(symbolic)):
			for j in range(i+1, len(symbolic)):
				if symbolic[i] == symbolic[j]:
					droped.append(j)
					print(symbolic[i], symbolic[j], i, j)
		sn = np.delete(symbolic, droped)
		nCandidatos = np.delete(mc, droped, 0)
		print(len(sn), nCandidatos.shape)
		return sn, nCandidatos

	def semp(self, psi, u, ym, nb, na, ni, rSymbol, index, delay, rho = 0.00001):
		pad = max(max(nb), max(na))
		y = ym[index, pad:]
		idx = np.arange(0, psi.shape[1])
		selected = []
		#print(idx)
		P = np.array([])
		Q = psi.copy()
	
		t = LSM(y, psi)
		Jold = np.inf#np.mean(np.square(y - (psi @ t)))
	
		#rho = 0.00001
		for i in range(ni):
			J = np.array([], np.float128)
	
			for j in range(Q.shape[1]):
				q = Q[:,j].reshape((-1,1))
				if i == 0:
					p = np.append(P, q).reshape((-1, 1))
				else:
					p = np.hstack((P,q))
				theta = (np.linalg.inv(p.T @ p) @ p.T) @ y
				#J = np.append(J, np.mean(np.square(y - (p @ theta))))
				sy = self.predict(u, ym, theta, rSymbol[selected + [idx[j]]], nb, na, index, delay)
				nj = np.mean(np.square(np.clip(y - sy[pad:], -10e6, 10e6)))
				J = np.append(J, nj)
				print(selected, idx[j], rSymbol[selected + [idx[j]]], nj) #, np.mean(np.square(y - (p @ theta)))
			J[np.isnan(J)] = np.Inf
			l = np.argmin(J)
			if J[l] < Jold and np.abs(J[l] - Jold) > rho:
				if P.shape[0] == 0:
					P = np.append(P, Q[:, l]).reshape((-1,1))
				else:
					P = np.hstack((P, Q[:, l].reshape((-1,1))))
				Q = np.delete(Q, l, 1)
				selected.append(idx[l])
				idx = np.delete(idx, l)
				print("adicionado", rSymbol[selected[-1]], J[l], Jold)
				Jold = J[l] #last
			else:
				print("não adicionado, fim", J[l], Jold)
				return P, selected
			
			#prunning
			flag = True
			print("prunning")
			while P.shape[1] > 1 and flag:
				Jp = np.array([])
				for k in range(P.shape[1]):
					R = np.delete(P, k, 1)
					theta = (np.linalg.inv(R.T @ R) @ R.T) @ y
					#Jp = np.append(Jp, np.mean(np.square(y - (R @ theta))))
					rt = selected.copy()
					rt.pop(k)
					sy = self.predict(u, ym, theta, rSymbol[rt], nb, na, index, delay)
					Jp = np.append(Jp, np.mean(np.square(y - sy[pad:])))
				m = np.argmin(Jp)
				if Jp[m] < Jold:
					P = np.delete(P, m, 1)
					selected.pop(m)
					Jold = Jp[m] #last
					continue
				else:
					flag = False #revisar
			#atualizando Jold
			theta = (np.linalg.inv(P.T @ P) @ P.T) @ y
			sy = self.predict(u, ym, theta, rSymbol[selected], nb, na, index, delay)
			Jold = np.mean(np.square(y - sy[pad:]))
			#Jold = np.mean(np.square(y - (P @ theta)))#J[l]
		return P, selected

	def frp(self, psi, y, ni, rho = 0.00001):
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
	
	def predict(self, u, y, theta, model, nb, na, index, delay=0, dt=0, ynonlinear=[0,0,0,0,0,0,0,0,0], unonlinear=[0,0,0,0,0,0,0,0,0], ymodifier=[0,0]):
		#Condição inicial
		#print("Simulação livre")
		d = max(max(na), max(nb))
		d = max(d, delay)
		d = max(d, max(ynonlinear))
		d = max(d, max(unonlinear))

		#print(d, y.shape, u.shape)
		yest = np.zeros(y.shape)
		yest[index, :d] = y[index, :d] #padding
		if y.shape[0] > 1:
			w = np.arange(0, y.shape[0], 1)
			w = np.delete(w, index)
			#print(index, w)
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


		for i in range(u.shape[0]):
			s += symbols('U'+str(i+1)+'.'+str(1)+':{}'.format(na[i]+1))
		
		#print('--------', s)
		iy = 0
		for k in range(d, y.shape[1]):
			num = np.array([])
			for i in range(y.shape[0]):
				num = np.hstack((num, np.flip(yest[i, k-nb[i]:k])))

			for i in range(u.shape[0]):
				num = np.hstack((num, np.flip(u[i, k-na[i]:k])))

			dicionario = dict(zip(s, num))
			aux = np.array([1 if m == 1 else m.evalf(subs=dicionario) for m in model])
			'''zz = symbols('Y2.1')
			if dicionario[zz] < 0:
				print(dicionario[zz], model, aux)'''
			try:
				#aux = np.array([1 if m == 1 else m.evalf(subs=dicionario) for m in model])
				#print(model, dicionario, aux)
				yest[index, k] = aux.real @ theta
			except ValueError:
				print("Error:")
				print(aux)
			except Exception as err:
				print(f"Unexpected {err=}, {type(err)=}")
				print(aux)
				
		return yest[index, :]
	
	def oneStepForward(self, u, y, theta, model, nb, na, index, dt=0, ymodifier=[0,0]):
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

		for i in range(u.shape[0]):
			s += symbols('U'+str(i+1)+'.1:{}'.format(na[i]+1))
		
		
		for k in range(d, y.shape[1]):
			num = np.array([])
			for i in range(y.shape[0]):
				num = np.hstack((num, np.flip(y[i, k-nb[i]:k])))

			for i in range(u.shape[0]):
				num = np.hstack((num, np.flip(u[i, k-na[i]:k])))

			dicionario = dict(zip(s, num))
			#print(dicionario)
			aux = np.array([1 if m == 1 else m.evalf(subs=dicionario) for m in model])
			#print(aux, dicionario, num.shape, len(s), nb, na)
			yest[index, k] = aux @ theta
		return yest[index, :]
	
	def oneStepForward2(self, u, y, theta, selected, nb, na, level, index, delay=0, dt=0, ynonlinear=[0,0,0,0,0,0,0,0,0], unonlinear=[0,0,0,0,0,0,0,0,0], ymodifier=[0,0]):
		#Condição inicial
		pad = max(max(nb), max(na))
		pad = max(pad, max(ynonlinear))
		pad = max(pad, max(unonlinear))

		#valCandidatos = self.matrix_candidate(u, y, nb, na, level, nonlinear, delay, dt, ymodifier)
		sv, valCandidatos = self.generate_candidate(u, y, nb, na, level, ynonlinear, unonlinear, False, delay, ymodifier)
		#print(valCandidatos.shape)
		Psi = valCandidatos[selected, :]

		yest = np.zeros(y.shape[1], np.float64)
		yest[:pad] = y[index, :pad]
		yest[pad:] = Psi.T @ theta
		return yest


#%%
'''
dataTank = pd.read_csv('data/coupletanks.csv')
u = np.reshape(np.array(dataTank['u']), (1,-1))
y = np.array(dataTank[['tank1', 'tank2']].T)

#Selecione o tanque 
output = 0  # 0 ou 1

num = [3, 5]
params = []
params.append({'nb':[1,0],'na':[1], 'level':2, 'ynonlinear':[0,0,0,0,0, 0,0,0,0,2, 0], 'unonlinear':[0,0,0,0,0, 0,0,0,0,0, 0], 'root':False, 'delay':0})
params.append({'nb':[1],'na':[2], 'level':2, 'ynonlinear':[0,0,0,0,0, 0,0,0,0,1, 0], 'unonlinear':[0,0,0,0,0, 0,0,0,0,0, 0], 'root':False, 'delay':0})

sselector = structureSelector()
ss = sselector.symbolic_regressors(**params[output])

vCandidatos = sselector.matrix_candidate(u, y, **params[output])
print(len(ss), vCandidatos.shape)
sym_regressors, num_regressors = sselector.generate_candidate(u, y, **params[output])
print(len(sym_regressors), num_regressors.shape)'''
'''
droped = []
for i in range(len(ss)):
	for j in range(i+1, len(ss)):
		if ss[i] == ss[j]:
			droped.append(j)
			print(ss[i], ss[j], i, j)
sn = np.delete(ss, droped)
nCandidatos = np.delete(vCandidatos, droped, 0)
print(len(sn), nCandidatos.shape, len(ss), vCandidatos.shape)'''
'''
plt.plot(vCandidatos[5])
plt.plot(vCandidatos[6])
plt.show()
pad = max(max(params[output]['nb']), max(params[output]['na']))
psi, selected  = sselector.semp(vCandidatos.T, u, y[:], params[output]['nb'], params[output]['na'], num[output], ss, output, params[output]['delay'], 0.000000001) #0.0000001
theta = LSM(y[output, pad:], psi)
model = ss[selected]
print(model, theta)'''
#%%
'''
slivre = sselector.predict(u, y, theta, ss[selected], params[output]['nb'], params[output]['na'], output, params[output]['delay'])
yhat = sselector.oneStepForward(u, y, theta, ss[selected], params[output]['nb'], params[output]['na'], output)
print("\nUm passo a frente")
metrics(y[output], yhat)
print("\nSimulação livre")
metrics(y[output], slivre)

l = 1.5
t = np.arange(0, y[0].shape[0], 1) * 0.1
plt.figure(figsize=(16, 6))
plt.plot(t, y[output].T, label='Sistema', linewidth=l)
plt.plot(t, yhat, label='um passo a frente', linewidth=l)
plt.plot(t, slivre, label='Livre', linewidth=l)
plt.margins(x=0.01)
plt.legend()
plt.tight_layout() 
plt.show()'''
#%%
'''
dt = 0.01
ui = np.reshape(u, (1, -1)).copy()
yi = np.reshape(y, (1, -1)).copy()

output = 0  
num = [6]
params = []
params.append({'nb':[2],'na':[2], 'level':2, 'nonlinear':[0,0,0,0,0,0,0,0,0,1], 'root':False, 'delay':1, 'diff':False, 'ymodifier':[0, 0]})

sselector = structureSelector()
#sselector.setLimits(-0.2, 0.2)
#clip.setLimit(-0.2, 0.2)
ss = sselector.symbolic_regressors(**params[output], intg=False)
print(ss)

vCandidatos = sselector.matrix_candidate(ui, yi.copy(), **params[output], dt=dt, intg=False)
print(len(ss), vCandidatos.shape)'''
'''
#plt.plot(vCandidatos[1])
plt.plot(vCandidatos[-4])
plt.plot(yi[0]**(3/4))
plt.show()

pad = max(max(params[output]['nb']), max(params[output]['na']))
psi, selected  = sselector.semp(vCandidatos.T, yi[output, pad:], num[output], 1e-13)

model = ss[selected]

theta = LSM(yi[output, pad:], psi)
print(model, theta, selected)

def metrics(y, yest):
    residuo1 = y - yest
    mape = round(np.mean(np.abs(residuo1 / (yest + np.finfo(np.float64).eps))), 5)
    plt.plot(np.abs(residuo1 / (yest + np.finfo(np.float64).eps)).T)
    print('RMSE:', np.sqrt(np.mean(np.square(residuo1))), 'MSE:', np.mean(np.square(residuo1)), '\nAET:', np.sum(np.abs(residuo1)), '\nMAPE:', str(mape) + '%')

model = ss[selected]
model[-1] = model[-1]*model[-1]**2
yhat1 = np.zeros(yi.shape[1])
yhat1[5:] = psi @ theta
yhat2 = sselector.oneStepForward(ui, yi, theta, model, params[output]['nb'], params[output]['na'], output, params[output]['diff'], dt=dt, intg=False)
print("\nUm passo a frente")
print(metrics(yi[0, 100:], yhat1[100:]))
print(metrics(yi[0, 100:], yhat2[100:]))
plt.show()'''

#%%
'''#my_data = np.genfromtxt('data/ballBeamTeste1.csv', delimiter=',')[1:,:]
my_data = np.genfromtxt('data/ballBeamNoise.csv', delimiter=',')[1:,:]
u = my_data[:, 0].copy()
y = my_data[:, 1].copy() 
t = my_data[:, -1].copy()

np.random.seed(15)
amplitude = 0.00001

dt = my_data[1, -1]'''
#%%
'''na = [2]
nb = [2]
level = 2
ss = structureSelector()
ss.setLimits(-1, 1)
d = 1
s = ss.symbolic_regressors(nb, na, level, nonlinear=[0,0,0,0,0, 0,0,0,0], root=False, diff=False, delay=d, intg=False, ymodifier=[1,0])
pprint(s)'''
#print(len(s), s[2].evalf(subs={"Y1.1":6}))
#%%
'''
na = [2]
nb = [1]
level = 1
ss = structureSelector()
#print(s[-1].evalf(subs={symbols('U1.1'):-3}))
t = np.arange(0,1000,1).reshape((1,-1))/10
u = np.cos(t)
#y = np.zeros((2,1000))
#y[0] = np.sin(t)
y = np.sin(t)
#y[1] = 2*u
dl = 1
v = ss.matrix_candidate(u, y, nb, na, level, nonlinear=[0,0,0,0,0, 0, 6], root=False, diff=False, delay=dl, intg=False, dt=0.1)
s = ss.symbolic_regressors(nb, na, level, nonlinear=[0,0,0,0,0, 0, 6], root=False, diff=False, delay=dl, intg=False)
pprint(s)
plt.plot(v.T)
plt.show()
print(v.shape, len(s))'''
#%%
'''plt.plot(v[2, :100].T)
plt.plot(v[5, :100].T)
plt.plot(v[7, :100].T)
plt.show()
print(v.shape)'''
#%%
'''
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


