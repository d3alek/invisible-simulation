""" Assumes fully observable sky. Predicts the yaw (on the spot rotation) from the degree and angle of polarization of samples from the sky
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import linear_model
from features.sky_model import SkyModelGenerator
from features.sun_calculator import sunrise_sunset, sun_position
import datetime
import random
from sklearn.model_selection import cross_val_score

import argparse

import pickle

from ast import literal_eval as make_tuple

import features.viewers as viewers

from matplotlib import pyplot as plt

DATA_FILE_NAME = 'data/latest_yaw_predictor_data.csv'

def angle_to_scalar(angle):
    """ Transforms angle to sin(angle) cos(ngle) pair to support good learning of periodic features"""
    return np.array([np.sin(angle), np.cos(angle)])

def to_series(name, sky_model):
    azimuths, altitudes = [*map(np.array, zip(*sky_model.observed_polar))]
    angles = np.array([*map(angle_to_scalar, sky_model.angles.flatten())])
    degrees = sky_model.degrees.flatten()

    #assert azimuths.shape == altitudes.shape and angles.shape == degrees.shape and azimuths.shape == degrees.shape

    polar_coordinates = list(zip(*map(np.rad2deg, [altitudes, azimuths])))

    angle_sin_series = pd.Series(angles[:,0], name=name, index=map(lambda a: 'As' + str(a), polar_coordinates))
    angle_cos_series = pd.Series(angles[:,1], name=name, index=map(lambda a: 'Ac' + str(a), polar_coordinates))
    degree_series = pd.Series(degrees, name=name, index=map(lambda d: 'D' + str(d), polar_coordinates))

    return angle_sin_series.append(angle_cos_series).append(degree_series)

def generate_data(date, days, hours, yaw_step):
    print("Generating data from %s for %d days at hours %s at %d degree step rotations" % (date, days, hours, np.rad2deg(yaw_step)))

    time_samples = []
    for hour in hours:
        time_samples.append(datetime.datetime.combine(date, datetime.time(hour, 0)))

    data = None

    yaw_range = np.arange(0, 2*np.pi, yaw_step)
    for yaw in yaw_range:
        print("Generating sky for yaw %s " % (np.rad2deg(yaw)), end="")
        for day in range(days):
            print(".", end="", flush=True)
            for time in time_samples:
                time = time + datetime.timedelta(days=day)
                sky_model = SkyModelGenerator(sun_position(time), yaw=yaw).generate(observed_polar = viewers.uniform_viewer())
                name = ' '.join(map(str, [time, yaw]))
                s = to_series(name, sky_model)
                s = s.append(pd.Series([time], name=name, index=['time']))
                s = s.append(pd.Series([yaw], name=name, index=['yaw']))

                if data is not None:
                    data = data.append(s)
                else: 
                    data = pd.DataFrame([s], index=[name])
        print()

    data.to_csv(DATA_FILE_NAME)
    print("Stored data")
    return data

def human_to_polar(string):
    return tuple(map(np.deg2rad, make_tuple(string[1:])))

def save_model(model):
    pickle.dump(model, open('data/yaw_classifier.pickle','wb'))

def analyze_data(data):
    yaws = data['yaw'].values 
    yaws_degrees = [*map(int, map(np.rad2deg, yaws))] # y
    exog = data.loc[:, data.columns[:-1]] # without yaw
    if exog.columns[-1] == "time": # without time
        exog = exog.loc[:, exog.columns[:-1]] 
    exog = exog.values # X
    clf = svm.LinearSVC()
    print("Fitting...", flush=True)
    print(clf.fit(exog, yaws_degrees))

    save_model(clf)

    return clf

def predict(classifier, datetime, yaw):
    sky_model = SkyModelGenerator(sun_position(datetime), yaw=yaw).generate(observed_polar=viewers.uniform_viewer())
    s = to_series(datetime, sky_model)
    s = s.values.reshape(1,-1)
    prediction_degrees = classifier.predict(s)[0]
    #print("Prediction for %s is %s should be %s" % (datetime, prediction_degrees, np.rad2deg(yaw)))
    return (yaw - np.deg2rad(prediction_degrees)) % (2*np.pi)

def plot_data(df, axis = 0):
    plt.figure(); df.mean(axis=axis).plot(); plt.legend(loc='best')
    plt.show()
    plt.figure(); df.median(axis=axis).plot(); plt.legend(loc='best')
    plt.show()

def parse_X(data):
    exog = data.loc[:, data.columns[:-1]] # without yaw
    if exog.columns[-1] == "time": # without time
        exog = exog.loc[:, exog.columns[:-1]] 
    exog = exog.values # X
    return exog

def parse_y(data):
    yaws = data['yaw'].values
    #yaws_degrees = np.array([*map(int, map(np.rad2deg, yaws))]) # y
    #return yaws_degrees
    return np.array([*map(angle_to_scalar, yaws)])

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('--date', default=today.strftime('%y%m%d'), help="start date for training data generation")
    parser.add_argument('--days', metavar='N', type=int, default=10,
                                help='the number of days to gather training data for')
    parser.add_argument('--hours', type=int, action='append',
                                help='sample the sky at this hour each day (can be specified multiple times)')
    parser.add_argument('--yaw-step', type=int, default=10,
                                help='rotational step in degrees')

    parser.add_argument('--load-training', action='store_true', default=False, help='load latest used data (default: calculate new data and save it as latest)')
    parser.add_argument('--training-samples', type=int, default=0, help='use N random samples from training data')
    parser.add_argument('--test', action='store_true', help='should the model be evaluated')
    parser.add_argument('--plot', action='store_true', help='should the evaluation be plotted')

    args = parser.parse_args()

    date = datetime.datetime.strptime(args.date, '%y%m%d').date()
    days = args.days
    hours = args.hours
    yaw_step_degrees = args.yaw_step
    load_training = args.load_training
    test = args.test
    plot = args.plot

    if load_training:
        data = pd.read_csv(DATA_FILE_NAME, index_col=0, parse_dates=True)
    else:
        data = generate_data(date, days, hours, np.deg2rad(yaw_step_degrees))

    if test:
        print("Cross validation...")
        clf = linear_model.MultiTaskLasso()
        #clf = svm.SVR()
        scores = cross_val_score(clf, parse_X(data), parse_y(data), cv=5, n_jobs=-1, scoring='neg_mean_absolute_error'); print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

