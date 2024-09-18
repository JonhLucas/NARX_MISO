import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import imageio

# Parâmetros do pêndulo
g = 9.81  # aceleração gravitacional (m/s^2)
L = 1.0   # comprimento da corda (m)
theta0 = np.pi / 4  # ângulo inicial (rad)
omega0 = 0.0  # velocidade angular inicial (rad/s)
tempo_total = 3  # tempo total de simulação (s)
fps = 30  # frames por segundo (para o GIF)

# Função que define a EDO do pêndulo
def pendulum_eq(y, t, g, L):
    theta, omega = y
    dydt = [omega, -(g/L) * np.sin(theta)]
    return dydt

# Condições iniciais
y0 = [theta0, omega0]

# Vetor de tempo para a simulação
t = np.linspace(0, tempo_total, fps * tempo_total)

# Resolver a EDO usando o método odeint
sol = odeint(pendulum_eq, y0, t, args=(g, L))

# Ângulo (theta) ao longo do tempo
theta = sol[:, 0]

# Criar o GIF
imagens = []
fig, ax = plt.subplots()

for i in range(len(t)):
    ax.clear()
    
    # Posição do pêndulo
    x = L * np.sin(theta[i])
    y = -L * np.cos(theta[i])
    
    # Desenhar o pêndulo
    ax.plot([0, x], [0, y], color='black', lw=2)  # linha da corda
    ax.plot(x, y, 'o', markersize=10, color='red')  # massa do pêndulo
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, 0.1)
    ax.set_aspect('equal')
    ax.set_title(f'Tempo: {t[i]:.2f} s')
    
    # Salvar o frame
    plt.draw()
    plt.pause(0.001)  # Para animação em tempo real (pode ser removido)
    
    # Converter a figura em array de imagem
    fig.canvas.draw()
    imagem = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    imagem = imagem.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imagens.append(imagem)

# Criar o GIF usando imageio
imageio.mimsave('pendulo_simples.gif', imagens, fps=fps)

plt.close()
