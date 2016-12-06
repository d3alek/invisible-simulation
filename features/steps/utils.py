from geometry import PolarPoint
import numpy as np

def convert_observed_and_sun_to_polar(context):
    observed_radians = PolarPoint.from_tuple(np.deg2rad(context.observed))
    sun_radians = PolarPoint.from_tuple(tuple(np.deg2rad(context.sun)))

    return observed_radians, sun_radians
