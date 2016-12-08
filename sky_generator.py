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

DATE_FORMAT = '%y%m%d'

def angle_to_scalar(angle):
    """ Transforms angle to sin(angle) cos(angle) as a good practice for learning on periodic features"""
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

def frequency_or_hours(frequency, hours):
    if frequency:
        return frequency
    else:
        return "-".join(hours)
def generate_data(file_name, datetimes, place, yaw_step):
    print("Generating skies for %d datetimes starting at %s and ending at %s at %d degree step rotations" %
            (len(datetimes), datetimes[0], datetimes[-1], np.rad2deg(yaw_step)))

    data = None

    yaw_range = np.arange(0, 2*np.pi, yaw_step)
    print("%d skies in total" % (len(yaw_range) * len(datetimes)))
    for yaw in yaw_range:
        print("Generating sky for yaw %s " % (np.rad2deg(yaw)), end="")
        for time in datetimes:
            print(".", end="", flush=True)
            sky_model = SkyModelGenerator.for_time_and_place(time, place, yaw).generate(viewers.uniform_viewer())
            name = ' '.join(map(str, [time, yaw]))
            s = to_series(name, sky_model)
            s = s.append(pd.Series([time], name=name, index=['time']))
            s = s.append(pd.Series([yaw], name=name, index=['yaw']))

            if data is not None:
                data = data.append(s)
            else: 
                data = pd.DataFrame([s], index=[name])
        print()

    return data

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('date', help="start date for training data generation. Format is 160501 for May 1 2016")
    parser.add_argument('--days', metavar='N', type=int, default=1, help='the number of days to gather training data for. Default 1')
    parser.add_argument('--start', type=int, default=6, help='the number of days to gather training data for. Default 6')
    parser.add_argument('--end', type=int, default=20, help='the number of days to gather training data for. Default 20')
    parser.add_argument('--freq', help='Mutually exclusive with --hours. Sample the sky between *start* hour and *end* hour at frequency - 10min, 1H, 1D')
    parser.add_argument('--hours', type=int, action='append',
                                help='Mutually exclusive with freq. Sample the sky at this hour each day (can be specified multiple times)')
    parser.add_argument('--yaw-step', type=int, default=10, help='rotational step in degrees. Default 10 degrees.')
    available_places = [item for item in dir(places) if not item.startswith("__") and not item in ['Place', 'PolarPoint', 'pytz']] 
    parser.add_argument('--place', default="sevilla", help='place on Earth to simulate the sky for. To add a place, edit places.py. Default sevilla. Available places: %s.' % available_places)


    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, DATE_FORMAT).date()
    days = args.days

    start_hour = args.start
    end_hour = args.end
    frequency = args.freq
    hours = args.hours
    assert frequency is None or hours is None, "Does not make sense to specify both frequency and hours."

    yaw_step_degrees = args.yaw_step
    place_name = args.place
    place = eval('places.%s' % place_name)

    file_name = 'skies/' + '-'.join([date.strftime(DATE_FORMAT), place_name, str(days), frequency_or_hours(frequency, hours), str(yaw_step_degrees)]) + '.csv'

    datetimes = []
    if frequency:
        start = datetime.datetime.combine(date, datetime.time(start_hour,0))
        end = datetime.datetime.combine(date, datetime.time(end_hour,0))
        one_day_range = pd.date_range(start=start, end=end, freq=frequency)
        for day in range(days):
            for time in one_day_range:
                datetimes.append(time + datetime.timedelta(days=day))
    else:
        assert hours is not None, "Must provide at least one of frequency or hours"
        for day in range(days):
            date_midnight = datetime.datetime.combine(date + datetime.timedelta(days=day), datetime.time(0,0))
            for hour in hours:
                datetimes.append(date_midnight + datetime.timedelta(hours=hour))

    data = generate_data(file_name, datetimes, place, np.deg2rad(yaw_step_degrees))

    data.to_csv(file_name)
    print("Stored data as csv: %s" % file_name)

