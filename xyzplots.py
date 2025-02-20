from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})

G = 6.67408e-11
m1 = 1.9885e30 # kg
m2 = 5.972e24 # kg
R = 1.49598e11 # m
radial_velocity = np.sqrt(G*(m1+m2)/R**3)

def LN(r):
    f1 = -G*m1/((r)*abs(r))
    f2 = -G*m2/((r-R)*abs(r-R))
    ang_v = G*r*(m1+m2)/(R**3)
    return ang_v + f1 + f2


L2 = root_scalar(LN, bracket=[1.50e11, 1.52e11]).root


def r1(x,y,z):
    return np.sqrt(x**2 + y**2 + z**2)


def r2(x,y,z):
    return np.sqrt((x-R)**2 + y**2 + z**2)


def EOM(t, X):
    Xdot = np.zeros(6)
    Xdot[:3] = X[3:6]
    x, y, z = X[:3]
    xdot, ydot = X[3:5]
    R1 = r1(x,y,z)
    R2 = r2(x,y,z)
    xddot = (x*radial_velocity**2 + 2*radial_velocity*ydot - G*m1*x/(R1**3)
        - G*m2*(x-R)/(R2**3))
    yddot = (y*radial_velocity**2 - 2*radial_velocity*xdot - G*m1*y/(R1**3)
        - G*m2*y/(R2**3))
    zddot = - G*m1*z/(R1**3) - G*m2*z/(R2**3)
    Xdot[3:6] = xddot, yddot, zddot
    return Xdot


def init_r(lpoint: float, pos: tuple):
    return np.array([lpoint+pos[0], pos[1], pos[2]])


def init_ydot(lpoint, pos: tuple) -> float:
    # magic constant is acquired through trial and error
    magic_constant = 2.2001182232956373
    r_0 = init_r(lpoint, pos)
    x = r_0[0]
    R1 = r1(*r_0)
    R2 = r2(*r_0)
    xddot = (x*radial_velocity**2 - G*m1*x/(R1**3)
        - G*m2*(x-R)/(R2**3))
    ydot = magic_constant*np.sqrt(abs(xddot) * abs(r1(*pos)))
    return ydot


def main():
    time_u = 24*3600 # time unit (days)
    duration = ... # days
    distance = 1000 # m


    def construct_3d_plot_L2(traj, pos: tuple) -> None:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(projection='3d')
        ax.plot(traj.y[0,...], traj.y[1,...], traj.y[2,...])
        ax.plot(L2, 0, 0, 'k+')

        r = init_r(L2, pos)
        ax.plot(*r, 'r+')
        ax.text(*r, 'P₀')

        #ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        bound = 3.5*abs(pos[0])
        ax.axes.set_xlim3d(left=L2 - bound, right=L2 + bound)
        ax.axes.set_ylim3d(bottom=-bound, top=bound)
        ax.axes.set_zlim3d(bottom=-bound, top=bound)
        ax.view_init(25, -140)

        plt.tight_layout()
        plt.show()


    def get_trajectory(dur, pos: tuple, iydot = None, lpoint = L2):
        # 'dur' argument is in days
        t = 24*3600 # time unit (days)
        r_0 = init_r(lpoint, pos) # m
        if iydot is None:
            ydot_0 = init_ydot(L2, pos) # r1 used for position vector
        else:
            ydot_0 = iydot
        init_ = np.array([*r_0, 0, ydot_0, 0])
        traj = solve_ivp(
            EOM,
            [0,dur*t],
            init_,
            atol=1e-6,
            rtol=3e-14
        )
        return traj


    dist = 1000

    # A special number for reference
    super_number = .000001305493871*dist

    traj1 = get_trajectory(45, (-1000,0,0), iydot=0)
    traj2 = get_trajectory(400, (-1000,0,0))
    traj3 = get_trajectory(400, (-2.5e8, 0, -3e8))
    construct_3d_plot_L2(traj1, (-1000,0,0))
    construct_3d_plot_L2(traj2, (-1000,0,0))
    construct_3d_plot_L2(traj3, (-2.5e8, 0, -3e8))


if __name__ == '__main__':
    main()
