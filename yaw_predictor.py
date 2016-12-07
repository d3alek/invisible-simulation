""" Predicts the yaw (on the spot rotation) from the degree and angle of polarization of samples from the sky
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sky_model import SkyModelGenerator
from sun_calculator import sun_position
import datetime
import sky_generator
import argparse
import pickle 
import viewers as viewers
from matplotlib import pyplot as plt

figure_rows_cols = {
        1: (1, 1), 
        2: (2, 1),
        3: (3, 1),
        4: (2, 2),
        5: (3, 2),
        6: (3, 2),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3)}

def save_model(model):
    pickle.dump(model, open('data/yaw_predictor_model.pickle','wb'))

def load_pickled_model():
    return pickle.load(open('data/yaw_predictor_model.pickle','rb'))

def angle_from_classifier_prediction(classifier_prediction, polar, decompose_yaw):
    if decompose_yaw:
        sin, cos = classifier_prediction
        angle = np.arctan2(sin, cos)
    else:
        angle = classifier_prediction
    
    if not polar:
        angle = angle % (2*np.pi)  # arctan2 returns in the range [-np.pi : np.pi] so we transform it to [0: 2*np.pi]

    return angle

def predict(classifier, datetime, yaw, polar, decompose_yaw, mask):
    sky_model = SkyModelGenerator(sun_position(datetime), yaw=yaw).generate(observed_polar=viewers.uniform_viewer())
    s = sky_generator.to_series(datetime, sky_model)
    s = s[s.index[mask]]
    assert not ('time' in s.index) and not ('yaw' in s.index) # only sin, cos and deg, should not include time and yaw
    s = s.values.reshape(1,-1)
    
    return angle_from_classifier_prediction(classifier.predict(s)[0], polar, decompose_yaw)

def parse_X(data):
    exog = data.loc[:, data.columns[:-1]] # without yaw
    if exog.columns[-1] == "time": # without time
        exog = exog.loc[:, exog.columns[:-1]] 
    exog = exog.values # X
    return exog

def parse_y(data, decompose_yaw):
    yaws = data['yaw']
    if decompose_yaw:
        return np.array([*map(sky_generator.angle_to_scalar, yaws.values)])
    else:
        return yaws

def rad_to_int(radians):
    return np.vectorize(int)(np.round(np.rad2deg(radians)))

def plot_expected_vs_actual(title, expected_yaws, actual_yaws, ax, polar):
    if not polar:
        actual_yaws = rad_to_int(actual_yaws)
        expected_yaws = rad_to_int(expected_yaws)
        ax.plot(expected_yaws, actual_yaws)
        ax.plot(expected_yaws, expected_yaws, color='r')
    else:
        ax.axes.get_yaxis().set_visible(False)
        ax.plot(actual_yaws, expected_yaws)
        ax.plot(expected_yaws, expected_yaws, color='r')

    ax.set_title(title)

def class_to_name(object):
    return str(object).split('(')[0]

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Do a linear regression on a sample of the sky over N days to predict the time of day.')
    parser.add_argument('--training', help="training dataset csv file path")
    parser.add_argument('--date', help="start date for training data generation")
    parser.add_argument('--days', type=int, default=1, help="number of days to run for (1 is default, means just the date)")
    parser.add_argument('--hours', type=int, action='append',
                                help='sample the sky at this hour each day (can be specified multiple times)')
    parser.add_argument('--yaw-step', type=int, default=10,
                                help='rotational step in degrees')
    parser.add_argument('--polar', action="store_true", help='produce polar plot')
    parser.add_argument('--load-model', action="store_true", help='load the model instead of training it')
    parser.add_argument('--decompose-yaw', action="store_true", help='split yaw into cos(yaw) sin(yaw) to capture cyclicity')
    parser.add_argument('--lowest-rank', type=int, default=0, help='use feature ranking (comes out of another script). 0 disables it (default), 1 means use only features ranked 1, etc.')

    args = parser.parse_args()

    training_file = args.training
    date = datetime.datetime.strptime(args.date, '%y%m%d').date()
    days = args.days
    hours = args.hours
    yaw_step_degrees = args.yaw_step
    polar = args.polar
    load_model = args.load_model
    decompose_yaw = args.decompose_yaw
    lowest_rank = args.lowest_rank

    data = pd.read_csv(training_file, index_col=0, parse_dates=True)

    if lowest_rank > 0:
        features_sin_rank_file = 'data/rfe_sin.pickle'
        features_cos_rank_file = 'data/rfe_cos.pickle'
        ranking_sin = np.append(pickle.load(open(features_sin_rank_file, 'rb')).ranking_, [1,1]) # to preserve the last 2 columns, time and yaw
        ranking_cos = np.append(pickle.load(open(features_cos_rank_file, 'rb')).ranking_, [1,1]) # to preserve the last 2 columns, time and yaw
        mask = np.ma.mask_or(ranking_sin <= lowest_rank, ranking_cos <= lowest_rank)
        data = data[np.arange(data.columns.size)[mask]]
        print("Only using features ranked <= %d so %d features selected" % (lowest_rank, data.columns.size - 2))
        mask = mask[:-2] # to remove the last 2 columns leaving only features in mask
    else:
        mask = np.full(data.columns.size - 2, True, dtype=bool) # to remove the last 2 columns, leaving only features in mask

    reg = linear_model.Ridge(alpha=1000)
    reg.fit(parse_X(data), parse_y(data, decompose_yaw))
    save_model(reg)

    print(reg)

    expected_yaws = np.deg2rad(range(0, 360, yaw_step_degrees))

    plots = len(hours) * days
    rows, cols = figure_rows_cols[plots]

    if polar:
        fig, axes = plt.subplots(subplot_kw=dict(polar=True, axisbg='none'), nrows=rows, ncols=cols)
    else:
        fig, axes = plt.subplots(nrows=rows, ncols=cols)

    flat_axes = np.array(axes).flatten()
    for day in range(days):
        date_midnight = datetime.datetime.combine(date + datetime.timedelta(days=day), datetime.time(0,0))
        for number, hour in enumerate(hours):
            actual_yaws = []
            time = date_midnight + datetime.timedelta(hours=hour)
            for expected_yaw in expected_yaws:
                actual_yaws.append(predict(reg, time, expected_yaw, polar, decompose_yaw, mask))
            assert len(expected_yaws) == len(actual_yaws)
            plot_number = day*len(hours) + number
            plot_expected_vs_actual(str(time), expected_yaws, np.array(actual_yaws), flat_axes[plot_number], polar)

    title = "%s on %s from %s for %d days %s %s yaw %s features" % (class_to_name(reg), training_file.split('/')[-1].split('.')[0], date, days, "polar" if polar else "", "sin cos" if decompose_yaw else "radians", data.columns.size - 2)

    fig.suptitle(title, size=16)

    fig.subplots_adjust(top=0.88)
    plt.savefig("graphs/" + title.replace(' ', '_') + '.png', dpi=fig.dpi)
    plt.show()

