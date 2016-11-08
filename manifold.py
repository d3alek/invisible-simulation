from sklearn import manifold
import datetime
import features.sun_calculator as sc
from features.sky_model import SkyModelGenerator as smg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from sklearn import preprocessing

parser = argparse.ArgumentParser(description='Display manifold of the polarized sky')
parser.add_argument('--start', default='160801 00:00')
parser.add_argument('--end', default='160802 00:00')
parser.add_argument('--freq', default="1H")
parser.add_argument('--yaw-to', type=int, default=1)
parser.add_argument('--yaw-step', type=int, default=36)
parser.add_argument('--method', default="mds")
parser.add_argument('--combine', action="store_true", default=False)

def extract_method(method_string):
    if method_string == 'mds':
        return manifold.MDS(n_components=2)
    if method_string == 'tsne':
        return manifold.TSNE(n_components=2)

args = parser.parse_args()
start = datetime.datetime.strptime(args.start, '%y%m%d %H:%M')
end = datetime.datetime.strptime(args.end, '%y%m%d %H:%M')
freq = args.freq
yaw_to = args.yaw_to
yaw_step = args.yaw_step
method = extract_method(args.method)
combine = args.combine

def sky_model(day, yaw_degrees):
        return smg(sc.sun_position(day), yaw=np.deg2rad(yaw)).generate()

yaw_range = range(0, yaw_to, yaw_step)
date_range = pd.date_range(start, end, freq=freq)
angles = pd.DataFrame()
degrees = pd.DataFrame()
cmap = plt.cm.jet(np.linspace(0,1,len(date_range)))
colors = []
for yaw in yaw_range:
    for num, date in enumerate(date_range):
        index = ','.join([date.strftime('%d.%m %Hh'), str(yaw)])
        angles.loc[:, index] = pd.Series(preprocessing.scale(sky_model(date, yaw).angles.flatten()))
        degrees.loc[:, index] = pd.Series(preprocessing.scale(sky_model(date, yaw).degrees.flatten()))
        colors.append(cmap[num])

angles_degrees = angles.append(degrees).T
angles = angles.T
degrees = degrees.T

assert angles_degrees.shape[0] == angles.shape[0]

if not combine:
    angles_projected = method.fit_transform(angles)
    degrees_projected = method.fit_transform(degrees)

    plt.figure()
    for index, (x, y) in enumerate(angles_projected):
        name = angles.index[index]
        plt.scatter([x], [y], s=100, c=colors[index])
        plt.text(x, y, name, color=(0,0,0), fontdict={'weight': 'bold', 'size': 10})

    plt.show()

    plt.figure()
    for index, (x, y) in enumerate(degrees_projected):
        name = degrees.index[index]
        plt.scatter([x], [y], s=100, c=colors[index])
        plt.text(x, y, name, color=(0,0,0), fontdict={'weight': 'bold', 'size': 10})

    plt.show()

else:
    together_projected = method.fit_transform(angles_degrees)
    plt.figure()
    for index, (x, y) in enumerate(together_projected):
        name = angles_degrees.index[index]
        plt.scatter([x], [y], s=100, c=colors[index])
        plt.text(x, y, name, color=(0,0,0), fontdict={'weight': 'bold', 'size': 10})

    plt.show()
