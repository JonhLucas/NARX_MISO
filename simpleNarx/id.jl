using DelimitedFiles, Plots
using ControlSystemIdentification, ControlSystemsBase

#%%
## Ball and beam
url = "https://ftp.esat.kuleuven.be/pub/SISTA/data/mechanical/ballbeam.dat.gz"
zipfilename = "/tmp/bb.dat.gz"
path = Base.download(url, zipfilename)
run(`gunzip -f $path`)
data = readdlm(path[1:end-3])
u = data[:, 1]' # beam angle
y = data[:, 2]' # ball position
d = iddata(y, u, 0.1)

dtrain = d[1:end÷2]
dval = d[end÷2:end]

# A model of order 2-3 is reasonable,
model,_ = newpem(dtrain, 3, stable=false)
sys = tf(model)
#%%
predplot(sys, dval, h=1)
predplot!(model, dval, h=10, ploty=false)
predplot!(model, dval, h=20, ploty=false)
simplot(sys, dval, ploty=false)
simplot!(sys, dval, ploty=false)
yhat = simulate(sys, dtrain)