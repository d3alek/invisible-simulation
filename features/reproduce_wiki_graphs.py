import sky_model
from sky_model import SkyModelGenerator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import ipdb

# Reproduces graph https://en.wikipedia.org/wiki/Rayleigh_sky_model#/media/File:Soldis_zendis.jpg

EAST = (0, np.pi/2)
WEST = (0, 3*np.pi/2)
sun = WEST # setting to the west

one_degree_in_radians = np.pi/180
observed_altitudes = np.arange(np.pi/2, step=one_degree_in_radians)
observed_azimuths = np.arange(2*np.pi, step=one_degree_in_radians)

def matrix_from_func(func):
    rows = []
    for observed_azimuth in  observed_azimuths:
        rows.append([func(observed_altitude, observed_azimuth) for observed_altitude in observed_altitudes])

    return np.array(rows)

gamma_func = lambda alt, azim: np.rad2deg(SkyModelGenerator().with_sun_at(sun).get_gamma((alt, azim)))
gamma_image = matrix_from_func(gamma_func)

theta_func = lambda alt, azim: np.rad2deg(SkyModelGenerator().with_sun_at(sun).get_theta((alt, azim)))
theta_image = matrix_from_func(theta_func)

images = [gamma_image, theta_image]

fig, axes = plt.subplots(nrows=2, ncols=1)
for ax, image in zip(axes.flat, images):
    im = ax.pcolor(image)
    ax.axis([0, 90, 0, 360])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.figure()

horizon_degrees = [SkyModelGenerator(max_degree=1).with_sun_at(sun).get_degree((0, azim)) for azim in observed_azimuths]
plt.plot(horizon_degrees);
plt.xlabel('Azimuth');
plt.ylabel('% Polarization on Horizon')

plt.show()


for sun in [EAST, (np.pi/2 - 0.8, np.pi), WEST]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sm = SkyModelGenerator().with_sun_at(sun).generate()

    u, v, w = np.empty(sm.angle_vectors.shape), np.empty(sm.angle_vectors.shape), np.empty(sm.angle_vectors.shape)

    for index, angle_vector in enumerate(sm.angle_vectors.flat):
            u.flat[index], v.flat[index], w.flat[index] = angle_vector

    ax.quiver(sm.x, sm.y, sm.z, u, v, w, length=0.1)

    ax.set_xlabel('x');
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    sun_x, sun_y, sun_z = sky_model.to_cartesian(sun)

    ax.scatter([sun_x],[sun_y],[sun_z],color="y",s=1000)

    plt.show()

