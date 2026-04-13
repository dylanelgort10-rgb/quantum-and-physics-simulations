import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----- space -----
N = 800
L = 1.0
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# ----- wave packet -----
x0 = L / 4
sigma = 0.05
k = 50

psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k * x)

# ----- time settings -----
dt = 0.001

# ----- plot setup -----
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2)

ax.set_ylim(0, 1)
ax.set_xlabel("Position")
ax.set_ylabel("Probability")
ax.set_title("Quantum Wave Packet Evolution")

# ----- update function -----
def update(frame):
    global psi

    laplacian = (np.roll(psi, 1) - 2 * psi + np.roll(psi, -1)) / dx**2
    psi = psi + 1j * 0.5 * laplacian * dt

    prob = np.abs(psi)**2
    prob /= prob.max()

    line.set_ydata(prob)
    return line,

# ----- animation -----
ani = FuncAnimation(fig, update, frames=400, interval=30)
plt.show()
