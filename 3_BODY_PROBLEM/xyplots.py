from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams.update({'font.size': 12})

G = 6.67408e-11
m1 = 1.9885e30 # kg
m2 = 5.972e24 # kg
R = 1.49598e8 # km

def LN(r): # This function defines the function defining the radial acceleration depending on the distance from the Sun
    #TODO: derive this equation mathematically (checked!)
    f1 = -G*m1/((r)*abs(r))
    f2 = -G*m2/((r-R)*abs(r-R))
    ang_v = G*r*(m1+m2)/(R**3)
    #ang_v = G*((m1+m2)*(x3+x1)-(m2*l))/(l**3)
    return ang_v + f1 + f2      # net radial acceleration

# ? LAGRANGE Points are the roots of the function LN
l1 = root_scalar(LN, bracket=[1.48e8, 1.49e8]).root
l2 = root_scalar(LN, bracket=[1.50e8, 1.52e8]).root
l3 = root_scalar(LN, bracket=[-1.52e8, -1.49e8]).root

# ? Compute L4 and L5 points
l4_x = R / 2
l4_y = R * np.sqrt(3) / 2
l5_x = R / 2
l5_y = -R * np.sqrt(3) / 2

# ? Print the distance between the Sun and the Lagrange points
print('Lagrange points relative distance to the sun (km):')
print('L1:', l1)
print('L2:', l2)
print('L3:', l3)
print('L4:', (l4_x, l4_y))
print('L5:', (l5_x, l5_y))

# ? Define the domain to plot properly
domain = np.arange(-2*R, 2*R, 100)
range_ = LN(domain)

f1 = plt.figure(figsize=(4.25, 4))
f2 = plt.figure(figsize=(4.25, 4))
f3 = plt.figure(figsize=(4.25, 4))
ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)
ax3 = f3.add_subplot(111)

ax_kwargs = {
    'xycoords' : 'data',
    'textcoords' : 'offset points',
    'horizontalalignment' : 'center',
    'verticalalignment' : 'top'
}

# ? Plotting of the Earth and Sun on the same plot with L points
ax1.plot(domain, range_)
ax1.grid()
ax1.set_ylim(-0.25e5, 0.25e5)
ax1.set_xlim(-abs(l3)*1.2, abs(l3*1.2))
ax1.plot(0, 0, 'oy')
ax1.plot(R, 0, 'og')
ax1.plot(l3, 0, '.k')
ax1.annotate('Sun', xy=(0,0), xytext=(0,15), **ax_kwargs)
ax1.annotate(
    'Earth',
    xy=(R,0),
    xycoords='data',
    xytext=(-2,15),
    textcoords='offset points',
    horizontalalignment='right',
    verticalalignment='top')
ax1.annotate('L3', xy=(l3,0), xytext=(0,15), **ax_kwargs)

# ? Focus on the Earth and L1 L2 points
ax2.plot(domain, range_)
ax2.grid()
ax2.set_xlim(l1/1.01, l2*1.01)
ax2.set_ylim(-5000, 5000)
ax2.plot(l1, 0, '.k')
ax2.plot(l2, 0, '.k')
ax2.plot(R, 0, 'og')
ax2.annotate('L1', xy=(l1,0), xytext=(0,15), **ax_kwargs)
ax2.annotate('L2', xy=(l2,0), xytext=(0,15), **ax_kwargs)
ax2.annotate('Earth', xy=(R,0), xytext=(0,15), **ax_kwargs)

# ? Plotting of the Earth and Sun on the same plot with L points and orbit
ax3.plot(domain, range_)
ax3.grid()
ax3.set_ylim(-abs(l3)*1.2, abs(l3*1.2))
ax3.set_xlim(-abs(l3)*1.2, abs(l3*1.2))
ax3.plot(0, 0, 'oy')
ax3.plot(R, 0, 'og')
ax3.plot(l3, 0, '.k')
ax3.annotate('Sun', xy=(0,0), xytext=(0,15), **ax_kwargs)
ax3.annotate(
    'Earth',
    xy=(R,0),
    xycoords='data',
    xytext=(-2,15),
    textcoords='offset points',
    horizontalalignment='right',
    verticalalignment='top')
ax3.annotate('L3', xy=(l3,0), xytext=(0,15), **ax_kwargs)
ax3.plot(l4_x, l4_y, '.k')
ax3.plot(l5_x, l5_y, '.k')
ax3.annotate('L4', xy=(l4_x, l4_y), xytext=(0,15), **ax_kwargs)
ax3.annotate('L5', xy=(l5_x, l5_y), xytext=(0,15), **ax_kwargs)

# * Add Earth's orbit as a circle on the second plot
earth_orbit = plt.Circle((0, 0), R, color='b', fill=False, linestyle='--')
ax3.add_artist(earth_orbit)

plt.show()
