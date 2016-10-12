import sky_model
from sky_model import SkyModelGenerator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import ipdb


EAST = (0, np.pi/2)
WEST = (0, 3*np.pi/2)
sun = WEST # setting to the west

one_degree_in_radians = np.pi/180
observed_altitudes = np.arange(np.pi/2, step=one_degree_in_radians)
observed_azimuths = np.arange(2*np.pi, step=one_degree_in_radians)

for sun in [EAST, (np.pi/2 - 0.8, np.pi), WEST]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sm = SkyModelGenerator().with_sun_at(sun).generate()

    u, v, w = np.empty(sm.angle_vectors.shape), np.empty(sm.angle_vectors.shape), np.empty(sm.angle_vectors.shape)
    linewidths = np.empty(sm.angle_vectors.size)

    for index, angle_vector in enumerate(sm.angle_vectors.flat):
            u.flat[index], v.flat[index], w.flat[index] = angle_vector
            linewidths[index] = 1 + 3*sm.degrees.flat[index]

    ax.quiver(sm.x, sm.y, sm.z, u, v, w, length=0.1, linewidths=linewidths, arrow_length_ratio = 0)

    ax.set_xlabel('south-north');
    ax.set_ylabel('west-east')
    ax.set_zlabel('altitude')

    sun_x, sun_y, sun_z = sky_model.to_cartesian(sun)

    ax.scatter([sun_x],[sun_y],[sun_z],color="y",s=1000)

    plt.show()

