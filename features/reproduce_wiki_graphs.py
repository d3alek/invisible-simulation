from sky_model import SkyModel
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import ipdb

# Reproduces graph https://en.wikipedia.org/wiki/Rayleigh_sky_model#/media/File:Soldis_zendis.jpg

sun = (0, 3*np.pi/2) # setting to the west

one_degree_in_radians = np.pi/180
observed_altitudes = np.arange(np.pi/2, step=one_degree_in_radians)
observed_azimuths = np.arange(2*np.pi, step=one_degree_in_radians)

def matrix_from_func(func):
    rows = []
    for observed_azimuth in  observed_azimuths:
        rows.append([func(observed_altitude, observed_azimuth) for observed_altitude in observed_altitudes])

    return np.array(rows)

#gamma_func = lambda alt, azim: np.rad2deg(SkyModel().with_sun_at(sun).get_gamma((alt, azim)))
#gamma_image = matrix_from_func(gamma_func)
#
#theta_func = lambda alt, azim: np.rad2deg(SkyModel().with_sun_at(sun).get_theta((alt, azim)))
#theta_image = matrix_from_func(theta_func)
#
#images = [gamma_image, theta_image]
#
#fig, axes = plt.subplots(nrows=2, ncols=1)
#for ax, image in zip(axes.flat, images):
#    im = ax.pcolor(image)
#    ax.axis([0, 90, 0, 360])
#
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(im, cax=cbar_ax)
#
#plt.figure()
#
#horizon_degrees = [SkyModel(max_degree=1).with_sun_at(sun).get_degree((0, azim)) for azim in observed_azimuths]
#plt.plot(horizon_degrees);
#plt.xlabel('Azimuth');
#plt.ylabel('% Polarization on Horizon')
#
#plt.show()
#

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

azimuths, altitudes = np.meshgrid(observed_azimuths, observed_altitudes)
x = np.cos(azimuths) * np.sin(altitudes)
y = np.sin(azimuths) * np.sin(altitudes)
z = np.cos(altitudes)

colors = np.empty(azimuths.shape)

for azimuth_index, azimuth in enumerate(observed_azimuths):
    for altitude_index, altitude in enumerate(observed_altitudes):
        colors[altitude_index, azimuth_index] = SkyModel().with_sun_at(sun).get_angle((altitude, azimuth)) 

normalized_colors = cm.hsv(colors/np.max(colors))
ax.plot_surface(x, y, z, facecolors = normalized_colors, alpha = 0.9, linewidth=0)

sun_x, sun_y, sun_z = np.sin(sun[1]), np.cos(sun[1]), np.sin(sun[0])
ax.scatter([sun_x],[sun_y],[sun_z],color="y",s=1000)

plt.show()

