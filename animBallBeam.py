
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
import matplotlib.animation as animation

my_data = np.genfromtxt('data/ballBeamTeste1.csv', delimiter=',')[1:,:]
u = my_data[:, 0].copy()
y = my_data[:, 1].copy() 
t = my_data[:, 3].copy()

dt = t[1]

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-0.21, 0.21), ylim=(-0.03, 0.03))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], '-', color='black', lw=2)
point = ax.scatter(1, 0, s=200)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def animate(i):
    lx = 0.2 * np.cos(u[i])
    ly = 0.2 * np.sin(u[i])

    thisx = [-lx, lx]
    thisy = [-ly, ly]

    px = (-y[i]) * np.cos(u[i])
    py = (-y[i]) * np.sin(u[i]) + 0.003
    
    line.set_data(thisx, thisy)
    #point.set_data(px, py)
    point.set_offsets((px, py))
    time_text.set_text(time_template % (i*dt))
    return line, point, time_text

print(dt)
ani = animation.FuncAnimation(
    fig, animate, y.shape[0], interval=dt*300, blit=True)
plt.show()