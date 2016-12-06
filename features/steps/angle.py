from behave import *
from sky_model import SkyModelGenerator
import numpy as np
from utils import convert_observed_and_sun_to_polar

@then('angle is {angle_descriptor}')
def step_impl(context, angle_descriptor):
    observed_radians, sun_radians = convert_observed_and_sun_to_polar(context)
    angle = np.round(np.rad2deg(SkyModelGenerator(sun_radians).get_angle(observed_radians)), 0)
    if angle_descriptor == "horizontal":
        assert angle%180 == 0, "Expected %s, actual %d" % (angle_descriptor, angle)
    elif angle_descriptor == "vertical":
        assert angle%90 == 0, "Expected %s, actual %d" % (angle_descriptor, angle)
    else: 
        assert angle == int(angle_descriptor), "Expected %s, actual %d" % (angle_descriptor, angle)
