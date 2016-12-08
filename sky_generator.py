""" Generates a data file containing the sky through different times
"""

import pandas as pd
import numpy as np
from sky_model import SkyModelGenerator
import datetime
import random
import argparse
import viewers
import places

DATA_FILE_NAME = 'data/latest_yaw_predictor_data.csv'

def angle_to_scalar(angle):
    """ Transforms angle to sin(angle) cos(ngle) pair to support good learning of periodic features"""
    return np.array([np.sin(angle), np.cos(angle)])

def to_series(name, sky_model):
    azimuths = np.array([*map(lambda a: a.azimuth, sky_model.observed_points)])
    altitudes = np.array([*map(lambda a: a.altitude, sky_model.observed_points)])
    angles = np.array([*map(angle_to_scalar, sky_model.angles.flatten())])
    degrees = sky_model.degrees.flatten()

    polar_coordinates = list(zip(*map(lambda r: np.round(np.rad2deg(r)), [altitudes, azimuths])))

    angle_sin_series = pd.Series(angles[:,0], name=name, index=map(lambda a: 'sin(%d,%d)' % (a[0], a[1]), polar_coordinates))
    angle_cos_series = pd.Series(angles[:,1], name=name, index=map(lambda a: 'cos(%d,%d)' % (a[0], a[1]), polar_coordinates))
    degree_series = pd.Series(degrees, name=name, index=map(lambda d: 'deg(%d,%d)' % (d[0], d[1]), polar_coordinates))

    return angle_sin_series.append(angle_cos_series).append(degree_series)

def generate_data(file_name, datetimes, yaw_step):
    print("Generating skies for %d datetimes starting at %s and ending at %s at %d degree step rotations" %
            (len(datetimes), datetimes[0], datetimes[-1], np.rad2deg(yaw_step)))

    data = None

    yaw_range = np.arange(0, 2*np.pi, yaw_step)
    print("%d skies in total" % (len(yaw_range) * len(datetimes)))
    for yaw in yaw_range:
        print("Generating sky for yaw %s " % (np.rad2deg(yaw)), end="")
        for time in datetimes:
            print(".", end="", flush=True)
            sky_model = SkyModelGenerator.for_time_and_place(time, places.sevilla, yaw).generate(viewers.uniform_viewer())
            name = ' '.join(map(str, [time, yaw]))
            s = to_series(name, sky_model)
            s = s.append(pd.Series([time], name=name, index=['time']))
            s = s.append(pd.Series([yaw], name=name, index=['yaw']))

            if data is not None:
                data = data.append(s)
            else: 
                data = pd.DataFrame([s], index=[name])
        print()

    data.to_csv(file_name)
    print("Stored data as csv: %s" % file_name)
    return data

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('--date', default="160501", help="start date for training data generation. Default May 1 2016")
    parser.add_argument('--days', metavar='N', type=int, default=1,
                                help='the number of days to gather training data for')
    parser.add_argument('--freq', default="10min", help='Mutually exclusive with --hours. Sample the sky at 6h and 20h at frequency - 10min, 1H, 1D')
    parser.add_argument('--yaw-step', type=int, default=10, help='rotational step in degrees')

    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%y%m%d').date()
    days = args.days
    frequency = args.freq
    yaw_step_degrees = args.yaw_step

    file_name = 'skies/' + '-'.join([date.strftime('%y%m%d'), str(days), frequency, str(yaw_step_degrees)]) + '.csv'
    start = datetime.datetime.combine(date, datetime.time(6,0))
    end = datetime.datetime.combine(date, datetime.time(20,0))
    one_day_range = pd.date_range(start=start, end=end, freq=frequency)
    datetimes = []
    for day in range(days):
        for time in one_day_range:
            datetimes.append(time + datetime.timedelta(days=day))

    data = generate_data(file_name, datetimes, np.deg2rad(yaw_step_degrees))
