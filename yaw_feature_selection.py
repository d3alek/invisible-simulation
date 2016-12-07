""" Selects the important features for yaw (on the spot rotation) from the degree and angle of polarization of samples from the sky
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import feature_selection
from sky_model import SkyModelGenerator
from sun_calculator import sun_position
import sky_generator
import datetime
import argparse
import pickle 
import viewers as viewers

def parse_X(data):
    exog = data.loc[:, data.columns[:-1]] # without yaw
    if exog.columns[-1] == "time": # without time
        exog = exog.loc[:, exog.columns[:-1]] 
    exog = exog.values # X
    return exog

def parse_y(data, decompose_yaw, component=1):
    yaws = data['yaw']
    if decompose_yaw:
        return np.array([*map(sky_generator.angle_to_scalar, yaws.values)])[:,component]
    else:
        return yaws

def class_to_name(object):
    return str(object).split('(')[0]

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('--training', help="training dataset csv file path")
    parser.add_argument('--date', help="start date for training data generation")
    parser.add_argument('--decompose-yaw', action="store_true", help='split yaw into cos(yaw) sin(yaw) to capture cyclicity')

    args = parser.parse_args()

    training_file = args.training
    date = datetime.datetime.strptime(args.date, '%y%m%d').date()
    decompose_yaw = args.decompose_yaw

    data = pd.read_csv(training_file, index_col=0, parse_dates=True)

    reg_sin = linear_model.Ridge(alpha=1000)
    rfe_sin = feature_selection.RFECV(estimator=reg_sin, verbose=True, n_jobs=-1)
    rfe_sin.fit(parse_X(data), parse_y(data, decompose_yaw, 0))
    pickle.dump(rfe_sin, open('data/rfe_sin.pickle', 'wb'))

    reg_cos = linear_model.Ridge(alpha=1000)
    rfe_cos = feature_selection.RFECV(estimator=reg_cos, verbose=True, n_jobs=-1)
    rfe_cos.fit(parse_X(data), parse_y(data, decompose_yaw, 1))
    pickle.dump(rfe_cos, open('data/rfe_cos.pickle', 'wb'))

    print("Saved pickled rfe_sin and rfe_cos in data directory") 
