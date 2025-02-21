from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.rcParams.update({'font.size': 12})

# Constantes physiques
G = 6.67408e-11
m1 = 1.9885e30  # kg (masse du Soleil)
m2 = 5.972e24  # kg (masse de la Terre)
R = 1.49598e11  # m (distance Terre-Soleil)
radial_velocity = np.sqrt(G * (m1 + m2) / R**3)  # Vitesse angulaire

# Fonction pour trouver le point de Lagrange L2
def LN(r):
    f1 = -G * m1 / (r * abs(r))
    f2 = -G * m2 / ((r - R) * abs(r - R))
    ang_v = G * r * (m1 + m2) / R**3
    return ang_v + f1 + f2

L2 = root_scalar(LN, bracket=[1.50e11, 1.52e11]).root  # Point de Lagrange L2

# Fonctions pour calculer les distances
def r1(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def r2(x, y, z):
    return np.sqrt((x - R)**2 + y**2 + z**2)

# Équations du mouvement
def EOM(t, X):
    Xdot = np.zeros(6)
    Xdot[:3] = X[3:6]  # Vitesses
    x, y, z = X[:3]
    xdot, ydot = X[3:5]
    R1 = r1(x, y, z)
    R2 = r2(x, y, z)
    xddot = (x * radial_velocity**2 + 2 * radial_velocity * ydot - G * m1 * x / (R1**3)
              - G * m2 * (x - R) / (R2**3))
    yddot = (y * radial_velocity**2 - 2 * radial_velocity * xdot - G * m1 * y / (R1**3)
              - G * m2 * y / (R2**3))
    zddot = -G * m1 * z / (R1**3) - G * m2 * z / (R2**3)
    Xdot[3:6] = xddot, yddot, zddot
    return Xdot

# Initialisation de la position
def init_r(lpoint: float, pos: tuple):
    return np.array([lpoint + pos[0], pos[1], pos[2]])

# Initialisation de la vitesse y
def init_ydot(lpoint, pos: tuple) -> float:
    r_0 = init_r(lpoint, pos)
    x = r_0[0]
    R1 = r1(*r_0)
    R2 = r2(*r_0)
    xddot = (x * radial_velocity**2 - G * m1 * x / (R1**3)
              - G * m2 * (x - R) / (R2**3))
    ydot = np.sqrt(abs(xddot) * abs(r1(*pos)))
    return ydot

# Calcul de la trajectoire
def get_trajectory(dur, pos: tuple, iydot=None, lpoint=L2):
    t = 24 * 3600  # Conversion des jours en secondes
    r_0 = init_r(lpoint, pos)
    if iydot is None:
        ydot_0 = init_ydot(L2, pos)
    else:
        ydot_0 = iydot
    init_ = np.array([*r_0, 0, ydot_0, 0])  # Conditions initiales
    traj = solve_ivp(
        EOM,
        [0, dur * t],
        init_,
        atol=1e-6,
        rtol=3e-14,
        t_eval=np.linspace(0, dur * t, 1000)  # Points de temps pour l'animation
    )
    return traj

# Animation de la trajectoire en 3D
def animate_trajectory_3d(traj):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    ax.plot(L2, 0, 0, 'k+', label='L2 Point')  # Point de Lagrange L2
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Calculer la taille maximale de la trajectoire
    max_range = np.max(np.abs(traj.y[:3, :])) / 1  # Taille maximale en x, y, z
    bound = 1.2 * max_range  # Ajouter une marge de 20%
    
    # Définir les limites des axes
    ax.set_xlim(L2 - bound / (10**8), L2 + bound / (10**8))
    ax.set_ylim(-bound / (10**8), bound / (10**8))
    ax.set_zlim(-bound / (10**8), bound / (10**8))
    
    # Initialisation de la ligne et du point
    line, = ax.plot([], [], [], 'r-', label='Trajectory')
    point, = ax.plot([], [], [], 'bo')
    
    # Fonction de mise à jour pour l'animation
    def update(frame):
        line.set_data(traj.y[0, :frame], traj.y[1, :frame])  # Mettre à jour x et y
        line.set_3d_properties(traj.y[2, :frame])           # Mettre à jour z
        point.set_data([traj.y[0, frame]], [traj.y[1, frame]])  # Mettre à jour x et y du point
        point.set_3d_properties([traj.y[2, frame]])             # Mettre à jour z du point
        return line, point
    
    # Création de l'animation
    ani = FuncAnimation(fig, update, frames=len(traj.t), interval=30, blit=False)
    plt.legend()
    plt.show()

# Animation de la trajectoire en 2D (plan X, Y)
def animate_trajectory_2d(traj):
    fig, ax = plt.subplots(figsize=(6, 5))  # Créer une figure 2D
    ax.plot(L2, 0, 'k+', label='L2 Point')  # Point de Lagrange L2
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Calculer la taille maximale de la trajectoire dans le plan X, Y
    max_range_x = np.max(np.abs(traj.y[0, :]))  # Taille maximale en x
    max_range_y = np.max(np.abs(traj.y[1, :]))  # Taille maximale en y
    bound_x = 1.2 * max_range_x  # Marge de 20% pour x
    bound_y = 1.2 * max_range_y*10**7  # Marge de 20% pour y
    
    # Définir les limites des axes
    ax.set_xlim(L2 - bound_x / (10**8), L2 + bound_x / (10**8))
    ax.set_ylim(-bound_y / (10**8), bound_y / (10**8))
    
    # Initialisation de la ligne et du point
    line, = ax.plot([], [], 'r-', label='Trajectory')
    point, = ax.plot([], [], 'bo')
    
    # Fonction de mise à jour pour l'animation
    def update(frame):
        line.set_data(traj.y[0, :frame], traj.y[1, :frame])  # Mettre à jour x et y
        point.set_data([traj.y[0, frame]], [traj.y[1, frame]])  # Mettre à jour x et y du point
        return line, point
    
    # Création de l'animation
    ani = FuncAnimation(fig, update, frames=len(traj.t), interval=30, blit=False)
    plt.legend()
    plt.show()

# Fonction principale
def main():
    # Calculer et animer les trajectoires
    traj = get_trajectory(100, (-1000, 0, 0))  # Trajectoire pour 100 jours
    animate_trajectory_3d(traj)  # Animation 3D
    animate_trajectory_2d(traj)   # Animation 2D
    
    traj2 = get_trajectory(400, (-1000, 0, 0))  # Trajectoire pour 400 jours
    animate_trajectory_3d(traj2)  # Animation 3D
    animate_trajectory_2d(traj2)   # Animation 2D
    
    traj3 = get_trajectory(400, (-2.5e11, 0, -3e11))  # Trajectoire pour 400 jours
    animate_trajectory_3d(traj3)  # Animation 3D
    animate_trajectory_2d(traj3)   # Animation 2D

if __name__ == '__main__':
    main()