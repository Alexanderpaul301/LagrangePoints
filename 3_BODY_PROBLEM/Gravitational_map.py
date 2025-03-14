import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-1
M_sun = 1.989e30  # mass of the Sun, kg
M_earth = 5.972e24  # mass of the Earth, kg
distance_sun_earth = 1.496e11  # average distance from Earth to Sun, meters

# Define the gravitational potential energy function
def gravitational_potential_energy(m1, m2, r):
    return -G * m1 * m2 / r

# Create a grid of distances
x = np.linspace(-2 * distance_sun_earth, 2 * distance_sun_earth, 500)
y = np.linspace(-2 * distance_sun_earth, 2 * distance_sun_earth, 500)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

# Compute the gravitational potential energy at each point in the grid
potential_energy = gravitational_potential_energy(M_sun, M_earth, R)

# Plot the gravitational potential energy
plt.figure(figsize=(10, 6))
contour = plt.contour(X, Y, potential_energy, levels=50, cmap='viridis')
plt.colorbar(contour, label='Potential Energy (J)')
plt.scatter(0, 0, color='yellow', s=100, label='Sun')  # Sun at the origin
plt.scatter(distance_sun_earth, 0, color='blue', s=50, label='Earth')  # Earth at its average distance from the Sun
plt.xlabel('X Distance (m)')
plt.ylabel('Y Distance (m)')
plt.title('Gravitational Potential Energy Map')
plt.legend()
plt.grid(True)
plt.show()