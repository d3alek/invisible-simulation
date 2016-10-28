""" Assumes fully observable sky. Predicts the time of day from the degree and angle of polarization of samples from the sky
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices
from features.sky_model import SkyModelGenerator
from sun_calculator import sunrise_sunset, sun_position
import datetime

import argparse

import pickle

from ast import literal_eval as make_tuple

store = pd.HDFStore('data/latest_polarization_time_predictor_data.h5')

ten_degrees_in_radians = np.pi/18
OBSERVED_ALTITUDES = np.arange(np.pi/2, step=ten_degrees_in_radians)
OBSERVED_AZIMUTHS = np.arange(2*np.pi, step=ten_degrees_in_radians)

def generate_data(days):
    EAST = (0, np.pi/2)
    sun_at = EAST

    day = datetime.datetime.utcnow()
    sunrise, sunset = sunrise_sunset(day)
    print("Sunrise: %s, Sunset %s" % (sunrise, sunset))
    day_length = sunset - sunrise
    hours = day_length.seconds/(3600)
    minutes = (day_length.seconds%(3600))/60
    print("Day length is %d:%02d hours" % (hours, minutes))

    time_samples = pd.date_range(start=sunrise, end=sunset, freq='H')

    data = None

    for day in range(days):
        for index, time in enumerate(time_samples):
            time = time + datetime.timedelta(days=day)
            sky_model = SkyModelGenerator(sun_position(time)).generate(observed_altitudes = OBSERVED_ALTITUDES, observed_azimuths = OBSERVED_AZIMUTHS)

            azimuths, altitudes = map(np.ndarray.flatten, np.meshgrid(sky_model.observed_azimuths, sky_model.observed_altitudes))
            angles = sky_model.angles.flatten()
            degrees = sky_model.degrees.flatten()

            assert azimuths.shape == altitudes.shape and angles.shape == degrees.shape and azimuths.shape == degrees.shape

            polar_coordinates = list(zip(*map(np.rad2deg, [altitudes, azimuths])))

            # TODO you're using dataframe as a series, use series instead.

            angle_df = pd.DataFrame(angles, index=map(lambda a: 'A' + str(a), polar_coordinates), columns=[time]).T
            degree_df = pd.DataFrame(degrees, index=map(lambda d: 'D' + str(d), polar_coordinates), columns=[time]).T

            df = angle_df.join(degree_df)
            df.loc[:, 'time'] = pd.Series([index], index=df.index)

            if data is not None:
                data = data.append(df)
            else: 
                data = df

    store['data'] = data
    return data

def human_to_polar(string):
    return tuple(map(np.deg2rad, make_tuple(string[1:])))

def analyze_data(data):
    endog = data['time']
    exog = data.loc[:, data.columns[:-1]]
    exog = sm.add_constant(exog)
    model = sm.OLS(endog, exog)
    results = model.fit()
    print(results.summary())

    params = results.params
    angles = params[1:params.size//2] # skipping constant column added by sm
    degrees = params[params.size//2+1:]

    result = {'angles': ([*map(human_to_polar, angles.index)], angles.values),'degrees': ([*map(human_to_polar, degrees.index)], degrees.values)}

    pickle.dump(result, open('data/polarization_time_predictor_result.pickle','wb'))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('--days', metavar='N', type=int, default=100,
                                help='the number of days to gather training data for')
    parser.add_argument('--load', action='store_true', default=False, help='load latest used data (default: calculate new data and save it as latest)')

    args = parser.parse_args()

    days = args.days
    load = args.load

    if load:
        data = store['data']
    else:
        data = generate_data(days)

    model = analyze_data(data)

    time = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    sky_model = SkyModelGenerator(sun_position(time)).generate(OBSERVED_ALTITUDES, OBSERVED_AZIMUTHS)
    angles = sky_model.angles.flatten()
    degrees = sky_model.degrees.flatten()
    angles_degrees = np.append(angles, degrees)
    angles_degrees_one = np.append([1], angles_degrees)
    print("Prediction for this time yesterday: %s" % model.predict(angles_degrees_one))
