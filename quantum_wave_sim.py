import numpy as np
import matplotlib.pyplot as plt

# ----- space -----
N = 800
L = 1.0
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# ----- initial wave packet -----
x0 = L / 4
sigma = 0.05
k = 50

psi = np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k * x)

# ----- simple time evolution -----
dt = 0.001
steps = 400

for _ in range(steps):
    laplacian = (np.roll(psi, 1) - 2 * psi + np.roll(psi, -1)) / dx**2
    psi = psi + 1j * 0.5 * laplacian * dt

# ----- plot result -----
prob = np.abs(psi)**2
prob /= prob.max()

plt.plot(x, prob)
plt.title("Quantum Wave Packet (Basic Simulation)")
plt.xlabel("Position")
plt.ylabel("Probability")
plt.show()
