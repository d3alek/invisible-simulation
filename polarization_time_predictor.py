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

OBSERVED_ALTITUDES = np.arange(np.pi/2, step=np.pi/20)
OBSERVED_AZIMUTHS = np.arange(2*np.pi, step=np.pi/5)

def generate_data(date, days, frequency_string, do_yaw):
    print("Generating data from %s for %d days at frequency %s" % (date, days, frequency_string))
    EAST = (0, np.pi/2)
    sun_at = EAST

    date = datetime.datetime.combine(date, datetime.time(10, 00)) # just to make a datetime, necessary for the following function to work
    sunrise, sunset = sunrise_sunset(date)
    print("Sunrise: %s, Sunset %s" % (sunrise, sunset))
    day_length = sunset - sunrise
    hours = day_length.seconds/(3600)
    minutes = (day_length.seconds%(3600))/60
    print("Day length is %d:%02d hours" % (hours, minutes))

    time_samples = pd.date_range(start=sunrise, end=sunset, freq=frequency_string)

    data = None

    if do_yaw:
        yaw_range = np.arange(0, 2*np.pi, np.pi/5)
    else:
        yaw_range = [0]
    for yaw in yaw_range:
        for day in range(days):
            print("Generating sky for day %s yaw %s" % (day, np.rad2deg(yaw)))
            for index, time in enumerate(time_samples):
                time = time + datetime.timedelta(days=day)
                sky_model = SkyModelGenerator(sun_position(time), yaw=yaw).generate(observed_altitudes = OBSERVED_ALTITUDES, observed_azimuths = OBSERVED_AZIMUTHS)

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
                df.loc[:, 'yaw'] = pd.Series([yaw], index=df.index)

                if data is not None:
                    data = data.append(df)
                else: 
                    data = df

    store['data'] = data
    print("Stored data")
    return data

def human_to_polar(string):
    return tuple(map(np.deg2rad, make_tuple(string[1:])))

def analyze_data(data):
    endog = data['time']
    exog = data.loc[:, data.columns[:-1]]
    if exog.columns[-1] == "time":
        exog = data.loc[:, data.columns[:-2]]
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

def predict(model, datetime, yaw=0):
    sky_model = SkyModelGenerator(sun_position(datetime), yaw=yaw).generate(OBSERVED_ALTITUDES, OBSERVED_AZIMUTHS)
    angles = sky_model.angles.flatten()
    degrees = sky_model.degrees.flatten()
    angles_degrees = np.append(angles, degrees)
    angles_degrees_one = np.append([1], angles_degrees)
    prediction = model.predict(angles_degrees_one)
    times = model.fittedvalues.index
    predicted_datetime = times[int(prediction)]
    predicted_time = predicted_datetime.time()
    time = datetime.time()
    if predicted_time > time:
        delta_time = datetime.combine(datetime, predicted_time) - datetime.combine(datetime, time)
    else:
        delta_time = datetime.combine(datetime, time) - datetime.combine(datetime, predicted_time)

    #print("For %s prediction %s which translates to %s (delta time %s)" % (datetime, prediction, predicted_datetime, delta_time))

    return delta_time

def timedelta_to_hours(timedelta):
    return timedelta.total_seconds()//3600

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('--date', default=today.strftime('%y%m%d'), help="start date for training data generation")
    parser.add_argument('--days', metavar='N', type=int, default=10,
                                help='the number of days to gather training data for')
    parser.add_argument('--freq', default="10mins",
                                help='how often to sample the time between sunset and sunrize (10mins, 1H, etc)')
    parser.add_argument('--load', action='store_true', default=False, help='load latest used data (default: calculate new data and save it as latest)')
    parser.add_argument('--yaw', default=False, help='should we include yaw in training')
    parser.add_argument('--test', default=True, help='should the model be evaluated')
    parser.add_argument('--test-date', default=today.strftime('%y%m%d'), help='date to start the test with')
    parser.add_argument('--test-days', default=10, type=int, help='how many days after the test date to test with')
    parser.add_argument('--test-time', default="10:00", help='time of day to start the test with')
    parser.add_argument('--test-minutes', default=300, type=int, help='minutes to test with each test day')

    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%y%m%d').date()
    days = args.days
    freq = args.freq
    load = args.load
    yaw = args.yaw
    test = args.test
    test_date = datetime.datetime.strptime(args.test_date, '%y%m%d').date()
    test_days = args.test_days
    test_time = datetime.datetime.strptime(args.test_time, '%H:%M').time()
    test_minutes = args.test_minutes

    if load:
        data = store['data']
    else:
        data = generate_data(date, days, freq, yaw)

    model = analyze_data(data)

    if test:
        print("Testing starting from %s %s for %d minutes each day" % (test_date, test_time, test_minutes))
        date = datetime.datetime.combine(test_date, test_time)
        df = pd.DataFrame()
        for days in range(test_days):
            prediction_errors = pd.Series()
            day = date + datetime.timedelta(days=days)
            for minutes in range(0, test_minutes, 30):
                yaw_errors = []
                day_time = day + datetime.timedelta(minutes=minutes)
                for yaw in np.arange(0, np.pi*2, 2*np.pi/5):
                    yaw_errors.append(predict(model, day_time, yaw))
                yaw_errors_hours = [*map(timedelta_to_hours, yaw_errors)]
                prediction_errors = prediction_errors.append(pd.Series(np.median(yaw_errors_hours), index=[day_time.time()]))
            df.loc[:, day.date()] = prediction_errors
            print ("%s error mean %s median %s" % (day.date(), np.mean(prediction_errors), np.median(prediction_errors)))

        df.to_csv('df.csv')
