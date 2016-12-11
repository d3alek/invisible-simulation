"""
Mostly based on https://en.wikipedia.org/wiki/Rayleigh_sky_model

The sky model is using the horizontal celestrial coordinate system https://en.wikipedia.org/wiki/Horizontal_coordinate_system 
this means we have two coordinates: 
 - altitude - elevation (0-pi/2)
 - azimuth - angle of the object around the horizon, from north increasing towards east
 - both in polar (radians)
"""

import numpy as np
from geometry import PolarPoint, unit_vector, angle_between, to_cartesian, rotate_yaw
from sun_calculator import sun_position

class SkyModelGenerator:
    zenith = PolarPoint(np.pi/2, 0)

    @classmethod
    def for_time_and_place(cls, time, place, yaw=0):
        obj = cls(sun_position(time, place), yaw)
        return obj

    def __init__(self, sun_position, yaw=0, max_degree=0.8):
        """ Contains the logic for calculating angle of polarization, degree of polarization 
            and intensity based on the position of the sun in the sky.

        Arguments: 
            sun_position - geometry.PolarPoint that describes the position of the sun in the sky
            yaw - angle in radians that describes the offset between the facing direction and true north (clockwise)
            max_degree - the maximum degree of polarization, in theory 1 but in nature observed no more than 0.8 (default)

        Use:
            As it is a generator, call the generate method passing an instance of Viewer
        """
        self.max_degree = max_degree
        self.yaw = yaw
        self.sun = self.to_local(sun_position) # where is the sun in our field of view

    def get_gamma(self, polar_point):
        """ Angular distance between the observed pointing and the sun. Scattering angle. """
        cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, polar_point])]
        return angle_between(cartesian_sun, cartesian_observed)

    def get_theta(self, polar_point):
        """ Only used in reproduce_wiki_graphs """
        return np.pi/2 - polar_point.altitude

    def get_degree(self, polar_point):
        gamma = self.get_gamma(polar_point)
        return self.max_degree * pow(np.sin(gamma), 2) / (1 + pow(np.cos(gamma), 2))

    def is_left_of_the_sun_antisun_split(self, polar_point):
        sun_azimuth = self.sun.azimuth
        normalized_sun_azimuth = self.sun.azimuth % (2*np.pi)
        observed_azimuth = polar_point.azimuth
        normalized_observed_azimuth = observed_azimuth % (2*np.pi)
        anti_sun_azimuth = (normalized_sun_azimuth + np.pi) % (2*np.pi)

        if anti_sun_azimuth > np.pi: 
            return normalized_observed_azimuth > normalized_sun_azimuth and normalized_observed_azimuth < anti_sun_azimuth
        else:
            return (normalized_observed_azimuth > normalized_sun_azimuth and normalized_observed_azimuth < 2*np.pi) or normalized_observed_azimuth < anti_sun_azimuth

    def get_angle(self, polar_point):
        cartesian_sun, cartesian_zenith, cartesian_observed = [*map(to_cartesian, [self.sun, self.zenith, polar_point])]

        # We get the vector perpendicular to the scattering plane formed by the observer, the zenith and the sun
        angle_vector = self.get_angle_vector(polar_point)
        # Get the normal of a vertical plane that goes through the zenith and the observed point
        vertical_plane_normal = np.cross(cartesian_zenith, cartesian_observed)
        # Calculate the angle from the observer's perspective by taking the angle between the vertical plane normal (acting as X axis) and the angle vector
        angle = angle_between(vertical_plane_normal, angle_vector) % (np.pi)
        # Take the reminder of division by pi as in reality we cannot distinguish whether angle is alpha or alpha+pi
         
        #TODO understand why this is necessary
        if self.is_left_of_the_sun_antisun_split(polar_point):
            angle = np.pi - angle

        return angle
    
    def get_angle_vector(self, polar_point):
        """ The vector perpendicular to the scattering plane formed by the observer, the zenith and the sun """
        cartesian_sun, cartesian_observed = [*map(to_cartesian, [self.sun, polar_point])]
        orthogonal = np.cross(cartesian_sun, cartesian_observed)

        # If vector lenght is close to 0, assume vector is horizontal. 
        # This happens when the sun is at the zenith.
        if np.isclose(np.linalg.norm(orthogonal), 0): 
            return (1, 0, 0)

        return orthogonal

    def generate(self, viewer):
        """ 
        Returns a sky model which assumes that the observer looks at *self.yaw*.
        The angles at the observed points are calibrated with this in mind. 
        If an observed point is at an angle *azimuth* from 0, the angle will be
        *angle0* - *azimuth*, where angle0 is calculated as if observer is looking at this point.

        The returned angles are always normalized in the range [0, pi].
        The returned degrees are in the range [0, self.max_degree].
        """
        angle_vectors = []
        angles = []
        degrees = []
        intensities = []

        observed_points = viewer.get_observed_points()
        for observed_point in observed_points:
            altitude, azimuth = observed_point.altitude, observed_point.azimuth
            angle_vectors.append(self.get_angle_vector(PolarPoint(altitude, azimuth)))

            # necessary to subtract azimuth because each point's angle is calculated 
            # as if observer is looking towards that point we assume that observer is looking at 0
            angle = (self.get_angle(PolarPoint(altitude, azimuth)) - azimuth) % np.pi
            assert angle >= 0 and angle <= np.pi, "Angle should be in [0,pi] but is %s" % angle
            angles.append(angle)
            degree = self.get_degree(PolarPoint(altitude, azimuth))
            assert degree >= 0 and degree <= self.max_degree, "Degree should be in [0,%s] but is %s" % (max_degree, degree)
            degrees.append(degree)
            intensity = self.get_intensity(PolarPoint(altitude, azimuth))
            intensities.append(intensity)

        cartesian = [*map(to_cartesian, observed_points)]
        x, y, z = [*map(np.array, zip(*cartesian))]
        return SkyModel(observed_points, x, y, z, np.array(angles), np.array(angle_vectors), np.array(degrees), np.array(intensities), self.yaw)

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

    # Table 1 from Perez et al, 4th data group, first row. Consult Perez et al 1993 for the effect of each parameter and suggested values
    #TODO Why choose these? Maybe change them to something Sevilla specific...
    clear_sky_parameters = (-1.4366, -0.1233, 1, 0.2809, 0.9938)

    def get_relative_luminance(self, zenith_angle, gamma):
        a, b, c, d, e = self.clear_sky_parameters
        return (1 + a * np.exp(b / np.cos(zenith_angle))) * (1 + c * np.exp(d*gamma) + e * np.cos(gamma)**2)

    def get_intensity(self, polar_point):
        """ Perez et al 1993 """
        # ratio between the luminance at the considered sky element and the luminance of an arbitrary reference sky element
        return self.get_relative_luminance(polar_point.altitude, self.get_gamma(polar_point))

class SkyModel:
    def __init__(self, observed_points, x, y, z, angles, angle_vectors, degrees, intensities, yaw):
        self.observed_points = observed_points
        self.x, self.y, self.z = x, y, z
        self.angles = angles
        self.angle_vectors = angle_vectors
        self.degrees = degrees
        self.intensities = intensities
        self.yaw = yaw

def test_sunset_sunrise():
    print("Testing whether sunset looks like sunrise when we yaw")
    from sun_calculator import sunrise_sunset, sun_position
    import datetime
    date = datetime.datetime.strptime("160801 08:00", '%y%m%d %H:%M')
    sunrise, sunset = sunrise_sunset(date)

    import viewers
    observed_polar = observed_polar=viewers.uniform_viewer()
    sunrise_angles = SkyModelGenerator(sun_position(sunrise)).generate(observed_polar).angles.flatten()

    for yaw in range(0, 360, 1):
        sunset_angles = SkyModelGenerator(sun_position(sunset), yaw=np.deg2rad(yaw)).generate(viewers.uniform_viewer()).angles.flatten()
        diff = np.median((sunset_angles - sunrise_angles) % np.pi)
        if diff < 0.1:
            print("Sunset looks like sunrise when we yaw %s degrees" % yaw)
            return True

    return False

if __name__ == "__main__":
    assert test_sunset_sunrise() == True, "Sunset did not look like sunrise as we rotated 360 degrees on the spot"
