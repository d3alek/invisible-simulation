import numpy as np
# using the horizontal celestrial coordinate system https://en.wikipedia.org/wiki/Horizontal_coordinate_system
# this means we have two coordinates: 
# - altitude - elevation (0-90)
# - azimuth - angle of the object around the horizon, from north increasing towards east

# To calculate the sun position changing during the day (altitude, azimuth), use https://en.wikipedia.org/wiki/Solar_zenith_angle https://en.wikipedia.org/wiki/Solar_azimuth_angle 
# or better yet, use https://pysolar.readthedocs.io/en/latest/ which takes longitude and latitude and time and gives you the altitude and azumith (take into account they take south as 0 for azimuth, with positive going towards east)

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
       print("It is possible you are passing degrees to a function that expects radians. Radians passed: %s", radians) 

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

    # angular distance between the observed pointing and the sun
    def get_gamma(self, point_radians):
       warn_if_looks_like_degrees(point_radians)
       cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, point_radians])]
       return angle_between(cartesian_sun, cartesian_observed)

    # the solar zenith distance (90\deg - solar altitude)
    def get_theta_sun(self):
       return np.pi/2 - self.sun[0]
    
    # the observed zenith distance (90\deg - observed altitude)
    def get_theta(self, point_radians):
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

    # http://www.math.ucla.edu/~ronmiech/Calculus_Problems/32A/chap11/section4/708d23/708_23.html
    def get_angle(self, point_radians):
        cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, point_radians])]
        orthogonal = np.cross(cartesian_sun, cartesian_observed)
        if np.isclose(np.linalg.norm(orthogonal), 0):
            return 0

        angle = np.arctan2(orthogonal[2], orthogonal[0])
        if angle < 0:
            angle += np.pi
        return angle
    
    # useful for 3d plotting
    def get_angle_vector(self, point_radians):
        cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, point_radians])]
        orthogonal = np.cross(cartesian_sun, cartesian_observed)
        if np.isclose(np.linalg.norm(orthogonal), 0):
            return (1, 0, 0)

        return unit_vector(orthogonal)

    def generate(self, observed_altitudes = DEFAULT_OBSERVED_ALTITUDES, observed_azimuths = DEFAULT_OBSERVED_AZIMUTHS):
        azimuths, altitudes = np.meshgrid(observed_azimuths, observed_altitudes)
        x = np.cos(azimuths) * np.sin(altitudes)
        y = np.sin(azimuths) * np.sin(altitudes)
        z = np.cos(altitudes)

        angle_vectors = np.empty(azimuths.shape, dtype=list)
        angles = np.empty(azimuths.shape)

        for azimuth_index, azimuth in enumerate(observed_azimuths):
            for altitude_index, altitude in enumerate(observed_altitudes):
                angle_vectors[altitude_index, azimuth_index] = self.get_angle_vector((altitude, azimuth)) 
                angles[altitude_index, azimuth_index] = self.get_angle((altitude, azimuth)) 

        return SkyModel(observed_azimuths, observed_altitudes, x, y, z, angles, angle_vectors)

class SkyModel:
    def __init__(self, observed_azimuths, observed_altitudes, x, y, z, angles, angle_vectors):
        self.observed_azimuths = observed_azimuths
        self.observed_altitudes = observed_altitudes
        self.x = x
        self.y = y
        self.z = z
        self.angles = angles
        self.angle_vectors = angle_vectors


if __name__ == "__main__":
    import doctest
    doctest.testmod()
