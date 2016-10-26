import numpy as np
import ipdb
# using the horizontal celestrial coordinate system https://en.wikipedia.org/wiki/Horizontal_coordinate_system
# this means we have two coordinates: 
# - altitude - elevation (0-90)
# - azimuth - angle of the object around the horizon, from north increasing towards east

seven_degrees_in_radians = 7*np.pi/180
DEFAULT_OBSERVED_ALTITUDES = np.arange(np.pi/2, step=seven_degrees_in_radians)
DEFAULT_OBSERVED_AZIMUTHS = np.arange(2*np.pi, step=seven_degrees_in_radians)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector) 

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> print(np.round(angle_between((1, 0, 0), (0, 1, 0)), 3))
            1.571
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> print(np.round(angle_between((1, 0, 0), (-1, 0, 0)), 3))
            3.142
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def round_print(a):
    to_print = np.round(a,1)
    to_print[to_print==0.] = 0. # to remove -0.
    return print(to_print)

def to_cartesian(spherical_coordinates):
    """ altitude goes (-90;90) in rest of the program so add 90\deg to it. 

    >>> round_print(to_cartesian((np.pi/2, 0)))
    [ 0.  0.  1.]
    >>> round_print(to_cartesian((-np.pi/2, 0)))
    [ 0.  0. -1.]
    >>> round_print(to_cartesian((0, np.pi/2)))
    [ 0.  1.  0.]
    >>> round_print(to_cartesian((0, 0)))
    [ 1.  0.  0.]
    """
    altitude = spherical_coordinates[0] + np.pi/2
    azimuth = spherical_coordinates[1]
    return np.array((np.sin(altitude) * np.cos(azimuth), np.sin(altitude) * np.sin(azimuth), -np.cos(altitude)))

def warn_if_looks_like_degrees(radians):
    if max(radians) > 2*np.pi:
       print("It is possible you are passing degrees to a function that expects radians. Radians passed: ", radians) 

def project(vector, plane_normal):
    """ Projects a vector onto a plane given by its normal
    http://www.euclideanspace.com/maths/geometry/elements/plane/lineOnPlane/

    >>> round_print(project([1, 1, 0], [0, 0, 1]))
    [ 1.  1.  0.]
    >>> round_print(project([1, 0, 0], [1, 0, 0]))
    [ 0.  0.  0.]
    >>> round_print(project([1, 1, 0], [1, 0, 0]))
    [ 0.  1.  0.]
    >>> round_print(project([1, 0, 0], [0, 1, 1]))
    [ 1.  0.  0.]
    """
    plane_normal = unit_vector(plane_normal)
    return np.cross(np.cross(plane_normal, vector), plane_normal)

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

class SkyModelGenerator:
    zenith = np.array((np.pi/2, 0))

    def __init__(self, max_degree = 0.8):
        self.max_degree = max_degree

    def with_sun_at_degrees(self, sun_position_degrees):
        radians = [*map(np.deg2rad,sun_position_degrees)]
        return self.with_sun_at(radians)

    def with_sun_at(self, sun_position_radians):
        warn_if_looks_like_degrees(sun_position_radians)
        self.sun = sun_position_radians
        return self

    def get_gamma(self, point_radians):
        """ Angular distance between the observed pointing and the sun. Scattering angle. """
        warn_if_looks_like_degrees(point_radians)
        cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, point_radians])]
        return angle_between(cartesian_sun, cartesian_observed)

    def get_theta_sun(self):
        """ The solar zenith distance (90\deg - solar altitude) """
        warn_if_looks_like_degrees(self.sun)
        return np.pi/2 - self.sun[0]
    
    def get_theta(self, point_radians):
        """ The observed zenith distance (90\deg - observed altitude) """
        warn_if_looks_like_degrees(point_radians)
        return np.pi/2 - point_radians[0]

    def get_degree(self, point_radians):
        warn_if_looks_like_degrees(point_radians)
        gamma = self.get_gamma(point_radians)
        return self.max_degree * pow(np.sin(gamma), 2) / (1 + pow(np.cos(gamma), 2))

    def get_phi(self, point_radians):
        warn_if_looks_like_degrees(point_radians)
        cartesian_sun, cartesian_zenith, cartesian_observed = [*map(to_cartesian, [self.sun, self.zenith, point_radians])]
        v1 = cartesian_sun - cartesian_zenith
        v2 = cartesian_observed - cartesian_zenith

        return angle_between(v1, v2)

    def get_angle(self, point_radians):
        # Knowing where the sun is, return an angle perpendicular to it
        cartesian_sun, cartesian_zenith, cartesian_observed = [*map(to_cartesian, [self.sun, self.zenith, point_radians])]
        #x, y = observed_to_sun_vector

        angle_vector = self.get_angle_vector(point_radians)

        return angle_between(cartesian_zenith, angle_vector)
    
    # useful for 3d plotting
    def get_angle_vector(self, point_radians):
        cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, point_radians])]
        orthogonal = np.cross(cartesian_sun, cartesian_observed)
        if np.isclose(np.linalg.norm(orthogonal), 0):
            return (1, 0, 0)

        return orthogonal

    def generate(self, observed_altitudes = DEFAULT_OBSERVED_ALTITUDES, observed_azimuths = DEFAULT_OBSERVED_AZIMUTHS):
        azimuths, altitudes = np.meshgrid(observed_azimuths, observed_altitudes)

        cartesian = map(to_cartesian, zip(altitudes, azimuths))

        angle_vectors = np.empty(azimuths.shape, dtype=list)
        angles = np.empty(azimuths.shape)
        degrees = np.empty(azimuths.shape)

        for azimuth_index, azimuth in enumerate(observed_azimuths):
            for altitude_index, altitude in enumerate(observed_altitudes):
                angle_vectors[altitude_index, azimuth_index] = self.get_angle_vector((altitude, azimuth)) 
                angles[altitude_index, azimuth_index] = self.get_angle((altitude, azimuth)) 
                degrees[altitude_index, azimuth_index] = self.get_degree((altitude, azimuth))

        x, y, z = zip(*cartesian)
        return SkyModel(observed_azimuths, observed_altitudes, x, y, z, angles, angle_vectors, degrees)

class SkyModel:
    def __init__(self, observed_azimuths, observed_altitudes, x, y, z, angles, angle_vectors, degrees):
        self.observed_azimuths = observed_azimuths
        self.observed_altitudes = observed_altitudes
        self.x = x
        self.y = y
        self.z = z
        self.angles = angles
        self.angle_vectors = angle_vectors
        self.degrees = degrees


if __name__ == "__main__":
    import doctest
    doctest.testmod()
