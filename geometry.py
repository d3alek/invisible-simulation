import numpy as np

def warn_if_looks_like_degrees(radians):
    if max(radians) > 4*np.pi:
       print("It is possible you are passing degrees to a function that expects radians: Radians passed: %d" % radians) 

class PolarPoint:
    def __init__(self, altitude, azimuth):
        warn_if_looks_like_degrees((altitude, azimuth))
        self.altitude = altitude
        self.azimuth = azimuth

    @classmethod
    def from_tuple(cls, tuple):
        obj = cls(tuple[0], tuple[1])
        return obj

    def __iter__(self):
        return iter([self.altitude, self.azimuth])

def unit_vector(vector):
    """ Returns the unit vector of the input vector.  """
    return vector / np.linalg.norm(vector) 

def angle_between(v1, v2):
    """ Returns the angle in radians between the input vectors::

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
    """ Formatter used for inline testing """
    to_print = np.round(a,1)
    to_print[to_print==0.] = 0. # to remove -0.
    return print(to_print)

def to_cartesian(local_polar):
    """ Altitude goes (-90;90) in rest of the program so add 90\deg to it. 

    >>> round_print(to_cartesian(PolarPoint(np.pi/2, 0)))
    [ 0.  0.  1.]
    >>> round_print(to_cartesian(PolarPoint(-np.pi/2, 0)))
    [ 0.  0. -1.]
    >>> round_print(to_cartesian(PolarPoint(0, np.pi/2)))
    [ 0.  1.  0.]
    >>> round_print(to_cartesian(PolarPoint(0, 0)))
    [ 1.  0.  0.]
    """
    altitude = local_polar.altitude + np.pi/2
    azimuth = local_polar.azimuth
    return np.array((np.sin(altitude) * np.cos(azimuth), np.sin(altitude) * np.sin(azimuth), -np.cos(altitude)))

def rotate_yaw(polar, yaw_radians):
    return PolarPoint(polar.altitude, polar.azimuth + yaw_radians)

if __name__ == "__main__":
    import doctest
    doctest.testmod()


