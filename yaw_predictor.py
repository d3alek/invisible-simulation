""" Predicts the yaw (on the spot rotation) from the degree and angle of polarization of samples from the sky
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from features.sky_model import SkyModelGenerator
from features.sun_calculator import sun_position
import datetime
import sky_generator
import argparse
import pickle
import features.viewers as viewers
from matplotlib import pyplot as plt

figure_rows_cols = {
        1: (1, 1), 
        2: (2, 1),
        3: (3, 1),
        4: (2, 2),
        5: (3, 2),
        6: (3, 2)}

def save_model(model):
    pickle.dump(model, open('data/yaw_classifier.pickle','wb'))

def predict(classifier, datetime, yaw):
    sky_model = SkyModelGenerator(sun_position(datetime), yaw=yaw).generate(observed_polar=viewers.uniform_viewer())
    s = sky_generator.to_series(datetime, sky_model) 
    assert not ('time' in s.index) and not ('yaw' in s.index) # only sin, cos and deg, should not include time and yaw
    s = s.values.reshape(1,-1)
    sin, cos = classifier.predict(s)[0]
    return np.arctan2(sin, cos) % (2*np.pi) # arctan2 returns in the range [-np.pi : np.pi] so we transform it to [0: 2*np.pi]

def parse_X(data):
    exog = data.loc[:, data.columns[:-1]] # without yaw
    if exog.columns[-1] == "time": # without time
        exog = exog.loc[:, exog.columns[:-1]] 
    exog = exog.values # X
    return exog

def parse_y(data):
    yaws = data['yaw'].values
    return np.array([*map(sky_generator.angle_to_scalar, yaws)])

def rad_to_int(radians):
    return np.vectorize(int)(np.round(np.rad2deg(radians)))

def plot_expected_vs_actual(title, expected_yaws, actual_yaws, ax):
    ax.plot(rad_to_int(expected_yaws), rad_to_int(actual_yaws))
    ax.set_xlabel('expected')
    ax.set_ylabel('actual')
    ax.set_title(title)

def class_to_name(object):
    return str(object).split('.')[-1].split('\'')[0].split('(')[0]

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('--training', help="training dataset csv file path")
    parser.add_argument('--date', help="start date for training data generation")
    parser.add_argument('--hours', type=int, action='append',
                                help='sample the sky at this hour each day (can be specified multiple times)')
    parser.add_argument('--yaw-step', type=int, default=10,
                                help='rotational step in degrees')

    args = parser.parse_args()

    training_file = args.training
    date = datetime.datetime.strptime(args.date, '%y%m%d').date()
    hours = args.hours
    yaw_step_degrees = args.yaw_step

    data = pd.read_csv(training_file, index_col=0, parse_dates=True)

    reg = linear_model.LinearRegression()

    reg.fit(parse_X(data), parse_y(data))

    expected_yaws = np.deg2rad(range(0, 360, yaw_step_degrees))

    date_midnight = datetime.datetime.combine(date, datetime.time(0,0))
    rows, cols = figure_rows_cols[len(hours)]
    plt.figure()
    fig, axes = plt.subplots(nrows=rows, ncols=cols)

    flat_axes = np.array(axes).flatten()
    for number, hour in enumerate(hours):
        actual_yaws = []
        time = date_midnight + datetime.timedelta(hours=hour)
        for expected_yaw in expected_yaws:
            actual_yaws.append(predict(reg, time, expected_yaw))
        assert len(expected_yaws) == len(actual_yaws)
        plot_expected_vs_actual(str(time), expected_yaws, np.array(actual_yaws), flat_axes[number])

    title = class_to_name(reg) + " on " + training_file.split('/')[-1].split('.')[0]
    fig.suptitle(title, size=16)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig("graphs/" + title.replace(' ', '_') + '.png')
    plt.show()

