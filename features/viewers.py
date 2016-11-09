import numpy as np

ten_degrees_in_radians = np.pi/18

def vertical_strip_viewer():
    observed_azimuths = np.arange(2*np.pi, step=ten_degrees_in_radians)
    def absolute_cosine(azimuths):
        """ 
        The start altitude is determined by the absolute value of the cosine of the azimuth - 
        the lower the value, the lower the start, end is always at zenith
        """
        points = []
        for azimuth in azimuths:
            start = np.abs(np.cos(azimuth))
            for altitude in np.arange(start, np.pi/2, step=ten_degrees_in_radians):
                points.append((altitude, azimuth))
        
        return np.array(points)

    return absolute_cosine(observed_azimuths)

def uniform_viewer():
    observed_azimuths = np.arange(2*np.pi, step=ten_degrees_in_radians)
    observed_altitudes = np.arange(np.pi/2, step = ten_degrees_in_radians)

    azimuths, altitudes = np.meshgrid(observed_azimuths, observed_altitudes)

    return [*zip(altitudes.flatten(), azimuths.flatten())]

