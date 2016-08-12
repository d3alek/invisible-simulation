from behave import *
from sky_model import SkyModel
import numpy as np

@then('angle is {angle_descriptor}')
def step_impl(context, angle_descriptor):
    observed_radians = [*map(np.deg2rad, context.observed)]
    angle = np.rad2deg(SkyModel().with_sun_at_degrees(context.sun).get_angle(observed_radians))
    if angle_descriptor == "horizontal":
        assert angle == 90, "Expected %s, actual %d" % (angle_descriptor, angle)
    elif angle_descriptor == "vertical":
        assert angle == 0, "Expected %s, actual %d" % (angle_descriptor, angle)
    else: 
        raise Exception("Undefined angle descriptor")
