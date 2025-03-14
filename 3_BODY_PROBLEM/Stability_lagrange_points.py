import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# Parameters for the three-body problem
mu = 3e-6                # Mass ratio of the two primary bodies # Approximation of the Sun-Earth system is mu=3*10^-6

L_points = {                    # Coordinates of the Lagrange points in a normalized frame
    'L1': (0.8369, 0),
    'L2': (1.155, 0),
    'L3': (-1.005, 0),
    'L4': (0.5, np.sqrt(3)/2),
    'L5': (0.5, -np.sqrt(3)/2)
}

# Function to compute the effective potential
def effective_potential(x, y, mu=mu):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    U = -0.5 * (x**2 + y**2) - (1 - mu) / r1 - mu / r2
    return U

# Derivatives of the potential
def omega_xx(x, y, mu=mu):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return 1 - (1 - mu) / r1**3 - mu / r2**3 + \
           3 * (1 - mu) * (x + mu)**2 / r1**5 + 3 * mu * (x - 1 + mu)**2 / r2**5

def omega_yy(x, y, mu=mu):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return 1 - (1 - mu) / r1**3 - mu / r2**3 + \
           3 * (1 - mu) * y**2 / r1**5 + 3 * mu * y**2 / r2**5

def omega_xy(x, y, mu=mu):
    r1 = np.sqrt((x + mu)**2 + y**2)
    r2 = np.sqrt((x - 1 + mu)**2 + y**2)
    return 3 * (1 - mu) * (x + mu) * y / r1**5 + 3 * mu * (x - 1 + mu) * y / r2**5

# Matrix A for stability analysis
def stability_matrix(x, y, mu=mu):
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [omega_xx(x, y, mu), omega_xy(x, y, mu), 0, 2],
        [omega_xy(x, y, mu), omega_yy(x, y, mu), -2, 0]
    ])
    return A

# Analyze stability of the Lagrange points
for point, (x, y) in L_points.items():
    A = stability_matrix(x, y, mu)
    eigenvalues = eig(A, right=False)
    stability = 'Stable' if np.all(np.real(eigenvalues) <= 0) else 'Unstable'
    print(f'Lagrange Point {point}: {stability}, Eigenvalues: {eigenvalues}')

# Plotting the effective potential
x_vals = np.linspace(-1.5, 1.5, 400)
y_vals = np.linspace(-1.5, 1.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
U = effective_potential(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, U, levels=50, cmap='viridis')

# Mark Lagrange points
for point, (x, y) in L_points.items():
    plt.plot(x, y, 'ro')
    plt.text(x, y, f'{point}', color='black', fontsize=12)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Effective Potential and Lagrange Points')
plt.colorbar(label='Potential')
plt.grid(True)
plt.show()
