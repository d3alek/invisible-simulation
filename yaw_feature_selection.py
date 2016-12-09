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

def parse_y(data, component=1):
    yaws = data['yaw']
    return np.array([*map(sky_generator.angle_to_scalar, yaws.values)])[:,component]

def class_to_name(object):
    return str(object).split('(')[0]

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Train a model and run feature ranking with recursive' +
            ' feature elimination and cross-validated selection (RFECV) of the best number of features')
    parser.add_argument('training', help="training dataset csv file path")

    args = parser.parse_args()

    training_file = args.training

    data = pd.read_csv(training_file, index_col=0, parse_dates=True)

    reg_sin = linear_model.Ridge(alpha=1000)
    rfe_sin = feature_selection.RFECV(estimator=reg_sin, verbose=True, n_jobs=-1)
    rfe_sin.fit(parse_X(data), parse_y(data, 0))
    pickle.dump(rfe_sin, open('data/rfe_sin.pickle', 'wb'))

    reg_cos = linear_model.Ridge(alpha=1000)
    rfe_cos = feature_selection.RFECV(estimator=reg_cos, verbose=True, n_jobs=-1)
    rfe_cos.fit(parse_X(data), parse_y(data, 1))
    pickle.dump(rfe_cos, open('data/rfe_cos.pickle', 'wb'))

    print("Saved pickled rfe_sin and rfe_cos in data directory") 
