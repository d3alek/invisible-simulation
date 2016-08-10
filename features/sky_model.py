import numpy as np
# using the horizontal celestrial coordinate system https://en.wikipedia.org/wiki/Horizontal_coordinate_system
# this means we have two coordinates: 
# - altitude - elevation (0-90)
# - azimuth - angle of the object around the horizon, from north increasing towards east

# To calculate the sun position changing during the day (altitude, azimuth), use https://en.wikipedia.org/wiki/Solar_zenith_angle https://en.wikipedia.org/wiki/Solar_azimuth_angle 
# or better yet, use https://pysolar.readthedocs.io/en/latest/ which takes longitude and latitude and time and gives you the altitude and azumith (take into account they take south as 0 for azimuth, with positive going towards east)

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
    return print(np.round(a,1))
def to_cartesian(spherical_coordinates):
    """ altitude goes (-90;90) in rest of the program so add 90\deg to it. 

    >>> round_print(np.round(to_cartesian((np.pi/2, 0)), 1))
    [ 0.  0. -1.]
    >>> round_print(to_cartesian((-np.pi/2, 0)))
    [ 0.  0.  1.]
    >>> round_print(to_cartesian((0, np.pi/2)))
    [ 0.  1.  0.]
    >>> round_print(to_cartesian((0, 0)))
    [ 1.  0.  0.]
    """
    altitude = spherical_coordinates[0] + np.pi/2
    azimuth = spherical_coordinates[1]
    return np.array((np.sin(altitude) * np.cos(azimuth), np.sin(altitude) * np.sin(azimuth), np.cos(altitude)))

def warn_if_looks_like_degrees(radians):
    if max(radians) > 2*np.pi:
       print("It is possible you are passing degrees to a function that expects radians. Radians passed: %s", radians) 

class SkyModel:
    zenith = np.array((np.pi/2, 0))
    antizenith = np.array((-np.pi/2, 0))

    def __init__(self):
        pass

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
       cartesian_sun, cartesian_antizenith, cartesian_observed = [*map(to_cartesian, [self.sun, self.antizenith, point_radians])]
       return angle_between(cartesian_sun - cartesian_antizenith, cartesian_observed - cartesian_antizenith)

    # the solar zenith distance (90\deg - solar altitude)
    def get_theta_sun(self):
       return np.pi/2 - self.sun[0]
    
    # the observed zenith distance (90\deg - observed altitude)
    def get_theta(self, point_radians):
       return np.pi/2 - point_radians[0]

# consult graph https://upload.wikimedia.org/wikipedia/commons/1/17/Rayleigh-geometry.pdf 
#gamma = 
#theta_sun 
#theta # the angular distance between the observed pointing and the zenith (90\deg - observed_altitude
#fi # angle between the zenith direction and the solar direction at the observed pointing
#tao # angle between the solar direction and the observed pointing at the zenith

#degree_max = 1
#degree = degree_max * sin(gamma)^2 / (1 + cos(gamma)^2)

#TODO reproduce the graphs in the wikipedia article https://en.wikipedia.org/wiki/Rayleigh_sky_model, starting with gamma's altitude-azimuth graph

if __name__ == "__main__":
    import doctest
    doctest.testmod()
