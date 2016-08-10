from behave import *
from sky_model import SkyModel
import numpy as np

@given('{sun_or_look} at the {direction}')
def step_impl(context, sun_or_look, direction):
    direction_to_degrees = {'horizon': 0, 'zenith': 90}
    direction_degrees = direction_to_degrees[direction]
    if 'look' in sun_or_look:
        context.observed = (direction_degrees, context.observed[1])

    else:
        context.sun = (direction_degrees, context.observed[1])

@given('the sun is at east')
def step_impl(context):
    context.sun = (context.sun[0], 90)

@given('we look at {direction}')
def step_impl(context, direction):
    degrees_from_direction = {'north': 0, 'east': 90, 'south': 180, 'west': 270}
    context.observed = (context.observed[0], degrees_from_direction[direction])
    if degrees_from_direction[direction] is None:
        raise Exception

@then('gamma is {expected_gamma:d}')
def step_impl(context, expected_gamma):
    observed_radians = [*map(np.deg2rad, context.observed)]
    gamma_degrees = np.floor(np.rad2deg(SkyModel().with_sun_at_degrees(context.sun).get_gamma(observed_radians)))
    assert np.isclose(gamma_degrees, expected_gamma), "Expected %d, actual %d" % (expected_gamma, gamma_degrees)

