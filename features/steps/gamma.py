from behave import *
from sky_model import SkyModelGenerator
import numpy as np
from utils import convert_observed_and_sun_to_polar

@given('{sun_or_look} at altitude {direction}')
def step_impl(context, sun_or_look, direction):
    direction_to_degrees = {'horizon': 0, 'zenith': 90}
    if direction.isdigit():
        direction_degrees = int(direction)
    else:
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

    observed_radians, sun_radians = convert_observed_and_sun_to_polar(context)
    gamma_degrees = np.floor(np.rad2deg(SkyModelGenerator(sun_radians).get_gamma(observed_radians)))
    assert np.isclose(gamma_degrees, expected_gamma), "Expected %d, actual %d" % (expected_gamma, gamma_degrees)

