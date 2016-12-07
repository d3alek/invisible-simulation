import sky_model
from sky_model import SkyModelGenerator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from geometry import PolarPoint
import viewers

EAST = (0, np.pi/2)
WEST = (0, 3*np.pi/2)
sun = WEST # setting to the west

one_degree_in_radians = np.pi/180
observed_altitudes = np.arange(np.pi/2, step=one_degree_in_radians)
observed_azimuths = np.arange(2*np.pi, step=one_degree_in_radians)

for sun in [EAST, (np.pi/2 - 0.8, np.pi), WEST]:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    sm = SkyModelGenerator(PolarPoint.from_tuple(sun)).generate(viewers.uniform_viewer())

    u, v, w = np.empty(len(sm.angle_vectors)), np.empty(len(sm.angle_vectors)), np.empty(len(sm.angle_vectors))
    linewidths = np.empty(len(sm.angle_vectors))

    for index, angle_vector in enumerate(sm.angle_vectors):
        u[index], v[index], w[index] = angle_vector
        linewidths[index] = 1 + 3*sm.degrees.flat[index]

    import ipdb
    ipdb.set_trace()

    ax.quiver(sm.x, sm.y, sm.z, u, v, w, length=0.1, linewidths=linewidths, arrow_length_ratio = 0)

    ax.set_xlabel('south-north');
    ax.set_ylabel('west-east')
    ax.set_zlabel('altitude')

    sun_x, sun_y, sun_z = sky_model.to_cartesian(PolarPoint.from_tuple(sun))

    ax.scatter([sun_x],[sun_y],[sun_z],color="y",s=1000)

    plt.show()

