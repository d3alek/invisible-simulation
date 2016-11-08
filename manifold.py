from sklearn import manifold
import datetime
import sun_calculator as sc
from features.sky_model import SkyModelGenerator as smg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Display manifold of the polarized sky')
parser.add_argument('--start', default='160801 00:00')
parser.add_argument('--end', default='160802 00:00')
parser.add_argument('--freq', default="1H")
parser.add_argument('--yaw-to', type=int, default=1)
parser.add_argument('--yaw-step', type=int, default=36)

args = parser.parse_args()
start = datetime.datetime.strptime(args.start, '%y%m%d %H:%M')
end = datetime.datetime.strptime(args.end, '%y%m%d %H:%M')
freq = args.freq
yaw_to = args.yaw_to
yaw_step = args.yaw_step

def sky_model(day, yaw_degrees):
        return smg(sc.sun_position(day), yaw=np.deg2rad(yaw)).generate()

yaw_range = range(0, yaw_to, yaw_step)
date_range = pd.date_range(start, end, freq=freq)
angles = pd.DataFrame()
cmap = plt.cm.jet(np.linspace(0,1,len(date_range)))
colors = []
for yaw in yaw_range:
    for num, date in enumerate(date_range):
        index = ','.join([date.strftime('%d.%m %Hh'), str(yaw)])
        angles.loc[:, index] = pd.Series(sky_model(date, yaw).angles.flatten())
        colors.append(cmap[num])

angles = angles.T

mds = manifold.MDS(n_components=2)
X_projected = mds.fit_transform(angles)

plt.figure()
for index, (x, y) in enumerate(X_projected):
    name = angles.index[index]
    plt.scatter([x], [y], s=100, c=colors[index])
    plt.text(x, y, name, color=(0,0,0), fontdict={'weight': 'bold', 'size': 10})

plt.show()
