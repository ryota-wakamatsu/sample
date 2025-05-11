
import numpy as np

def heat_diffusion_1d(nx=100, nt=100, alpha=0.01, dt=0.001, initial_peak=100):
    dx = 1.0 / (nx - 1)
    u = np.zeros(nx)
    u[nx // 2] = initial_peak
    history = [u.copy()]
    for _ in range(nt):
        unew = u.copy()
        for i in range(1, nx - 1):
            unew[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
        u = unew
        history.append(u.copy())
    return np.linspace(0, 1, nx), history

def heat_diffusion_1d_vectorized(nx=100, nt=100, alpha=0.01, dt=0.001, initial_peak=100):
    dx = 1.0 / (nx - 1)
    u = np.zeros(nx)
    u[nx // 2] = initial_peak
    history = [u.copy()]
    for _ in range(nt):
        unew = u.copy()
        unew[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[2:] - 2*u[1:-1] + u[:-2])
        u = unew
        history.append(u.copy())
    return np.linspace(0, 1, nx), history
