from sklearn import manifold
import datetime
import features.sun_calculator as sc
from features.sky_model import SkyModelGenerator as smg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from sklearn import preprocessing

from mpl_toolkits.mplot3d import Axes3D

import features.viewers as viewers

parser = argparse.ArgumentParser(description='Display manifold of the polarized sky')
parser.add_argument('--start', default='160801 00:00')
parser.add_argument('--end', default='160802 00:00')
parser.add_argument('--freq', default="1H")
parser.add_argument('--yaw-to', type=int, default=1)
parser.add_argument('--yaw-step', type=int, default=36)
parser.add_argument('--method', default="mds")
parser.add_argument('--combine', action="store_true", default=False)
parser.add_argument('--dimensions', type=int, default=2)
parser.add_argument('--const', type=float, default=0.2)

args = parser.parse_args()
start = datetime.datetime.strptime(args.start, '%y%m%d %H:%M')
end = datetime.datetime.strptime(args.end, '%y%m%d %H:%M')
freq = args.freq
yaw_to = args.yaw_to
yaw_step = args.yaw_step
combine = args.combine
dimensions = args.dimensions
const = args.const

def extract_method(method_string):
    if method_string == 'mds':
        return manifold.MDS(n_components=dimensions)
    if method_string == 'tsne':
        return manifold.TSNE(n_components=dimensions)

method = extract_method(args.method)

def sky_model(day, yaw_degrees):
        return smg(sc.sun_position(day), yaw=np.deg2rad(yaw)).generate(observed_polar=viewers.vertical_strip_viewer())

yaw_range = range(0, yaw_to, yaw_step)
date_range = pd.date_range(start, end, freq=freq)
angles = pd.DataFrame()
degrees = pd.DataFrame()
cmap = plt.cm.jet(np.linspace(0,1,len(date_range)))
colors = []
for yaw_num, yaw in enumerate(yaw_range):
    for num, date in enumerate(date_range):
        index = ','.join([date.strftime('%d.%m %Hh'), str(yaw)])
        angles.loc[:, index] = pd.Series(sky_model(date, yaw).angles.flatten())
        degrees.loc[:, index] = pd.Series(sky_model(date, yaw).degrees.flatten())
        colors.append(cmap[num])

names = angles.T.index

angles_degrees = angles.T
angles_degrees[degrees.T<const] = 0
angles_degrees = preprocessing.scale(angles_degrees)

angles = preprocessing.scale(angles.T)
degrees = preprocessing.scale(degrees.T)

assert angles_degrees.shape[0] == angles.shape[0]

def plot(df, names):
    fig = plt.figure()
    if dimensions == 2:
        ax = fig.add_subplot(111)
    if dimensions == 3:
        ax = fig.add_subplot(111, projection='3d')

    for index, data in enumerate(df):
        name = names[index]
        if dimensions == 2:
            x, y = data
            ax.scatter([x], [y], s=100, c=colors[index])
            ax.text(x, y, name, color=(0,0,0), fontdict={'weight': 'bold', 'size': 10})
        if dimensions == 3:
            x, y, z = data
            ax.scatter([x], [y], [z], s=100, c=colors[index])
            ax.text(x, y, z, name, color=(0,0,0), fontdict={'weight': 'bold', 'size': 10})

    plt.show()

if not combine:
    angles_projected = method.fit_transform(angles)
    degrees_projected = method.fit_transform(degrees)

    plot(angles_projected, names)
    plot(degrees_projected, names)

else:
    together_projected = method.fit_transform(angles_degrees)
    plot(together_projected, names)
