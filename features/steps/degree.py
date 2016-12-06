from behave import *
from sky_model import SkyModelGenerator
import numpy as np
from utils import convert_observed_and_sun_to_polar

@then('degree is {expected_degree:d}')
def step_impl(context, expected_degree):
    expected_degree_normalized = expected_degree / 100
    observed_radians, sun_radians = convert_observed_and_sun_to_polar(context)
    degree = np.round(SkyModelGenerator(sun_radians).get_degree(observed_radians), 2)
    assert np.isclose(degree, expected_degree_normalized), "Expected %f, actual %f" % (expected_degree_normalized, degree)

