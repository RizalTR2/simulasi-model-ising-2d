import numpy as np
import matplotlib.pyplot as plt
import random

def metropolis_step(grid, T):
    N = grid.shape[0]
    x, y = random.randint(0, N-1), random.randint(0, N-1)
    s_neighbors = (
        grid[(x+1)%N, y] + grid[(x-1)%N, y] +
        grid[x, (y+1)%N] + grid[x, (y-1)%N]
    )
    delta_E = 2 * grid[x, y] * s_neighbors
    if delta_E < 0 or random.random() < np.exp(-delta_E / T):
        grid[x, y] *= -1
    return grid

def run_simulation(N=20, temp=1.0, n_steps=100000):
    grid = np.random.choice([-1, 1], size=(N, N))
    magnetization_history = []
    for step in range(n_steps):
        grid = metropolis_step(grid, temp)
        if step % 100 == 0:
            magnetization = np.mean(grid)
            magnetization_history.append(magnetization)
    return grid, magnetization_history

if __name__ == "__main__":
    grid_size = 20
    temperatures = 4.0
    monte_carlo_steps = 200000
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle('2D Ising Model Simulation via Metropolis Algorithm')
    final_grid, M_history = run_simulation(N=grid_size, temp=temperatures, n_steps=monte_carlo_steps)

    ax_grid = axes[0]
    ax_grid.imshow(final_grid, cmap='binary', vmin=-1, vmax=1)
    ax_grid.set_title(f"Final State at T = {temperatures:.2f}")

    ax_mag = axes[1]
    ax_mag.plot(M_history)
    ax_mag.set_title(f"Magnetization at T = {temperatures:.2f}")
    ax_mag.set_xlabel("Monte Carlo Steps (x100)")
    ax_mag.set_ylabel("Average Magnetization")
    ax_mag.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.show()