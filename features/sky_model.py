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
    if max(radians) > 4*np.pi:
       print("It is possible you are passing degrees to a function that expects radians. Radians passed: ", radians) 

def rotate_yaw(polar, yaw_radians):
    return (polar[0], polar[1] + yaw_radians)

class SkyModelGenerator:
    zenith = np.array((np.pi/2, 0))

    def __init__(self, sun_radians, max_degree=0.8, yaw=0):
        warn_if_looks_like_degrees(sun_radians)
        self.sun = sun_radians
        self.max_degree = max_degree
        self.yaw = yaw

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
        local_point_radians = self.to_local(point_radians)
        gamma = self.get_gamma(local_point_radians)
        return self.max_degree * pow(np.sin(gamma), 2) / (1 + pow(np.cos(gamma), 2))

    def is_left_of_the_sun_antisun_split(self, observed_radians):
        sun_azimuth = self.sun[1]
        observed_azimuth = observed_radians[1]
        anti_sun_azimuth = (sun_azimuth + np.pi) % (2*np.pi)

        if anti_sun_azimuth > np.pi: 
            return observed_azimuth > sun_azimuth and observed_azimuth < anti_sun_azimuth
        else:
            return (observed_azimuth > sun_azimuth and observed_azimuth < 2*np.pi) or observed_azimuth < anti_sun_azimuth

    def get_angle(self, point_radians):
        local_point_radians = self.to_local(point_radians)
        cartesian_sun, cartesian_zenith, cartesian_observed = [*map(to_cartesian, [self.sun, self.zenith, local_point_radians])]

        angle_vector = self.get_angle_vector(point_radians)
        vertical_plane_normal = np.cross(cartesian_zenith, cartesian_observed)

        angle = angle_between(vertical_plane_normal, angle_vector)
        
        #TODO understand why this is necessary
        if self.is_left_of_the_sun_antisun_split(local_point_radians):
            angle = -angle
       
        return self.to_world(angle)
    
    # useful for 3d plotting
    def get_angle_vector(self, point_radians):
        local_point_radians = self.to_local(point_radians)
        cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, local_point_radians])]
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

    def get_sun_polar(self):
        return self.to_world(self.sun)

    def to_world(self, polar):
        if np.isscalar(polar):
            return polar + self.yaw
        else:
            return rotate_yaw(polar, self.yaw)
    
    def to_local(self, polar):
        if np.isscalar(polar):
            return polar - self.yaw
        else:
            return rotate_yaw(polar, -self.yaw)

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
