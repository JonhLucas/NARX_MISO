import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
def symbolic_candidatesGS(nb, na, level):
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
  #print(li)
  for j in range(level-1):
    lx = []
    k = 0
    for i in range(size):
      k += t[j, i]
      #print(li[i], aux[int(k-1):])
      lx = lx + (list(np.array(aux[int(k-1):]) * li[i]))
    #print(lx)
    l1 = np.hstack((l1, lx))
    aux = lx
  M = l1
  return M
def matrix_candidateGS(u, y, nb, na, level):
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
        #print(i,l)
        for j in range(level-1):
            lx = []
            k = 0

            for v in range(size):
                k += t[j, v]
                lx.extend(np.array(aux[int(k-1):]) * li[v])

            l1 = np.hstack((l1, lx))
            aux = lx

        M.append(l1)
    M = np.array(M)
    return M

def gram_schmidt(Psi,y,limit=2):
    reg = np.array([],dtype=np.uint8)
    cols = np.array([])
    ERRS = np.array([])
    #Definindo a primeira coluna da matriz final
    for i in range(Psi.shape[1]):
        wi = Psi[:,i]
        gi = np.dot(wi.T,y)/np.dot(wi.T,wi)
        ERRS = np.append(ERRS,(gi**2* np.dot(wi.T,wi))/np.dot(y.T,y))
    arg1 = np.argmax(ERRS)
    reg = np.append(reg,arg1)
    cols = np.append(cols,Psi[:,arg1]).reshape((Psi.shape[0],1))
    while(len(reg) < limit):
        i = 0
        ERRS = np.array([])
        max_err = -np.inf
        argmax = -1
        wks = []
        #Ortogonalizando as colunas
        while(i<limit):
            if np.all(i!=reg):
                pi = Psi[:,i]
                su = 0
                for j in range(len(reg)):
                    wj = Psi[:,j]
                    alphaj = np.dot(wj.T,pi)/np.dot(wj.T,wj)
                    su += alphaj*wj
                wk = pi - su
                wks += [wk.reshape(wk.shape[0],1)]
                gk = np.dot(wk.T,y)/np.dot(wk.T,wk)
                ERR = gk**2 * (np.dot(wk.T,wk)/np.dot(y.T,y))
                ERRS = np.append(ERRS,ERR)
                if ERR> max_err:
                    max_err = ERR
                    argmax = i
            i+=1
        #Selecionando a melhor coluna ortogonalizada
        cols = np.hstack((cols,wks[np.argmax(ERRS)]))
        reg = np.append(reg,argmax)
        
    return cols,reg
def callGram(y,u,ny,nu,l,limit):
    Psi = matrix_candidateGS(u,y,ny,nu,l)
    PsiSym = symbolic_candidatesGS(nu,ny,l)
    p,reg = gram_schmidt(Psi,y[ny:],limit)
    s = []
    for x in reg:
        s += [PsiSym[x]]
    return p,s,reg

#Realizando o processo de ortogonalização da matriz de regressores gerada a partir dos dados de validação
def ortogonalize(Psi,reg):
    Ort = Psi[:,0].reshape((Psi.shape[0],1))
    for i in range(1,len(reg)):
        pi = Psi[:,reg[i]]
        su = np.zeros(pi.shape)
        for j in range(i):
            wj = Psi[:,j]
            alphaj = np.dot(wj.T,pi)/np.dot(wj.T,wj)
            su += alphaj*wj
        wk = pi - su
        Ort = np.hstack((Ort,wk.reshape(wk.shape[0],1)))
    return Ort

def getParams(y,u,ny,nu):
    y_half = y[:y.shape[0]//4+ny]
    y_half2 = y[y.shape[0]//4-ny:]
    u_half = u[:u.shape[0]//4+ny]
    u_half2 = u[u.shape[0]//4-nu:]
    return y_half,y_half2,u_half,u_half2


def readFile(filename):
    with open(filename,"r+") as f:
        u = []
        y = []
        for line in f:
            a2i,a3i = [np.float64(x) for x in line.split()]
            u.append(a2i)
            y.append(a3i)
    return np.asarray(y),np.asarray(u)
def doPlot(y_hat1,y_hat2,y_half,y_half2):
    plt.plot(y_hat1,'b')
    plt.plot(y_half[ny:],'r')
    plt.title("1/4")
    plt.legend(["Real","Est"])
    plt.show()
    print("1/4")
    #EAT
    print(np.sum(np.abs(y_half[ny:]-y_hat1)))
    #print("MAPE")
    print(np.mean((y_half[ny:]-y_hat1)/y_half[ny:]))

    plt.plot(y_hat2,'b')
    plt.plot(y_half2[ny:],'r')
    plt.legend(["Real","Est"])
    plt.title('3/4')

    plt.show()
    print("3/4:")
    #EAT
    print(np.sum(np.abs(y_half2[ny:]-y_hat2)))
    #print("MAPE")
    #print(np.mean((y_half2[ny:]-y_hat2)/y_half2[ny:]))
def gramPlot(y_h,y_h2,u_h,u_h2,lim,ny,nu,l):
    p,s,r = callGram(y_h,u_h,ny,nu,l,limit=lim)
    print(p)
    print(s)
    print(r)
    print(s,'\t',r)
    theta = (np.linalg.inv(p.T@p)@p.T)@y_h[ny:]
    print(theta,'\n')
    Psi1 = matrix_candidateGS(u_h,y_h,ny,nu,l)
    p1 = ortogonalize(Psi1,r)
    Psi2 = matrix_candidateGS(u_h2,y_h2,ny,nu,l)
    p2 = ortogonalize(Psi2,r)
    y_hat1 = p1@theta
    y_hat2 = p2@theta
    doPlot(y_hat1,y_hat2,y_h,y_h2)

def doIdentification(filename,lim,ny,nu,l):
    y,u = readFile(filename)
    y_h,y_h2,u_h,u_h2 = getParams(y,u,ny,nu)
    print("New example\n")
    gramPlot(y_h,y_h2,u_h,u_h2,lim,ny,nu,l)
