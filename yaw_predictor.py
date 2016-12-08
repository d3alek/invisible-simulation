""" Predicts the yaw (on the spot rotation) from the degree and angle of polarization of samples from the sky
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sky_model import SkyModelGenerator
import datetime
import sky_generator
import argparse
import pickle 
import viewers as viewers
from matplotlib import pyplot as plt
import places

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

def angle_from_prediction(prediction, polar):
    sin, cos = prediction 
    angle = np.arctan2(sin, cos)
    
    if not polar:
        angle = angle % (2*np.pi)  # arctan2 returns in the range [-np.pi : np.pi] so we transform it to [0: 2*np.pi]

    return angle

def predict(predictor, series, polar, mask):
    s = series[series.index[mask]]
    assert not ('time' in s.index) and not ('yaw' in s.index) # only sin, cos and deg, should not include time and yaw
    s = s.values.reshape(1,-1)
    
    return angle_from_prediction(predictor.predict(s)[0], polar)

def parse_X(data):
    exog = data.loc[:, data.columns[:-1]] # without yaw
    if exog.columns[-1] == "time": # without time
        exog = exog.loc[:, exog.columns[:-1]] 
    exog = exog.values # X
    return exog

def parse_y(data):
    yaws = data['yaw']
    return np.array([*map(sky_generator.angle_to_scalar, yaws.values)])

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

def file_name(file_path):
    return file_path.split('/')[-1].split('.')[0]

if __name__ == "__main__":
    today = datetime.datetime.utcnow().date()
    parser = argparse.ArgumentParser(description='Train a linear regression on the sky and evaluate the resulting model.')
    parser.add_argument('training', help="training dataset csv file path")
    parser.add_argument('test', help="test dataset csv file path")
    parser.add_argument('--polar', action="store_true", help='produce polar plot')
    parser.add_argument('--use-time', action="store_true", help='use time as a feature')
    parser.add_argument('--lowest-rank', type=int, default=0, help='use feature ranking (comes out of another script). 0 disables it (default), 1 means use only features ranked 1, etc.')

    args = parser.parse_args()

    training_file = args.training
    test_file = args.test
    polar = args.polar
    lowest_rank = args.lowest_rank

    training_data = pd.read_csv(training_file, index_col=0, parse_dates=True)
    test_data = pd.read_csv(test_file, index_col=0, parse_dates=True)

    # TODO refactor this, maybe need to crop test as well as training
    #if lowest_rank > 0:
    #    features_sin_rank_file = 'data/rfe_sin.pickle'
    #    features_cos_rank_file = 'data/rfe_cos.pickle'
    #    ranking_sin = np.append(pickle.load(open(features_sin_rank_file, 'rb')).ranking_, [1,1]) # to preserve the last 2 columns, time and yaw
    #    ranking_cos = np.append(pickle.load(open(features_cos_rank_file, 'rb')).ranking_, [1,1]) # to preserve the last 2 columns, time and yaw
    #    mask = np.ma.mask_or(ranking_sin <= lowest_rank, ranking_cos <= lowest_rank)
    #    data = data[np.arange(data.columns.size)[mask]]
    #    print("Only using features ranked <= %d so %d features selected" % (lowest_rank, data.columns.size - 2))
    #    mask = mask[:-2] # to remove the last 2 columns leaving only features in mask
    #else:
    #    mask = np.full(data.columns.size - 2, True, dtype=bool) # to remove the last 2 columns, leaving only features in mask
    mask = np.full(training_data.columns.size - 2, True, dtype=bool) # to remove the last 2 columns, leaving only features in mask

    reg = linear_model.Ridge(alpha=1000)
    reg.fit(parse_X(training_data), parse_y(training_data))

    print(reg)

    first_time = ' '.join(test_data.index[0].split(' ')[0:2])
    yaws_for_a_time = test_data[test_data.time==first_time].shape[0]

    plots = test_data.shape[0] / yaws_for_a_time
    rows, cols = figure_rows_cols[plots]

    if polar:
        fig, axes = plt.subplots(subplot_kw=dict(polar=True, axisbg='none'), nrows=rows, ncols=cols)
    else:
        fig, axes = plt.subplots(nrows=rows, ncols=cols)

    flat_axes = np.array(axes).flatten()

    # This predicted dataframe will have times as rows, expected yaws as columns and predicted yaws as values
    predicted_df = pd.DataFrame()

    sorted = test_data.sort_values(by=['time', 'yaw'])
    for _, row in sorted.iterrows():
        expected = row.yaw
        predicted = predict(reg, row, polar, mask)
        time = row.time
        predicted_df.at[time, expected] = predicted

    # Plot expected vs predicted for each test time
    expected = predicted_df.columns
    for plot_number, (time, row) in enumerate(predicted_df.iterrows()):
        predicted = row.values
        plot_expected_vs_actual(time, expected, predicted, flat_axes[plot_number], polar)

    title = "%s on training[%s] test[%s] %s %s features" % (class_to_name(reg), file_name(training_file), file_name(test_file), "polar" if polar else "", training_data.columns.size - 2)

    fig.suptitle(title, size=16)

    # Save figure
    fig.subplots_adjust(top=0.88)
    save_path = "graphs/" + title.replace(' ', '_') + '.png'
    plt.savefig(save_path, dpi=fig.dpi)
    print("Saved figure to %s" % save_path)

    # Show figure
    plt.show()

