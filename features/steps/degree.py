from behave import *
from sky_model import SkyModel
import numpy as np

@then('degree is {expected_degree:d}')
def step_impl(context, expected_degree):
    expected_degree_normalized = expected_degree / 100
    observed_radians = [*map(np.deg2rad, context.observed)]
    degree = np.round(SkyModel().with_sun_at_degrees(context.sun).get_degree(observed_radians),2)
    assert np.isclose(degree, expected_degree_normalized), "Expected %f, actual %f" % (expected_degree_normalized, degree)
