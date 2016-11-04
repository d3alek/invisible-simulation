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

args = parser.parse_args()
start = datetime.datetime.strptime(args.start, '%y%m%d %H:%M')
end = datetime.datetime.strptime(args.end, '%y%m%d %H:%M')
freq = args.freq

def sky_model_for_day(day):
        return smg(sc.sun_position(day), yaw=0).generate()

date_range = pd.date_range(start, end, freq=freq)
angles = pd.DataFrame()
for date in date_range:
    angles.loc[:, date] = pd.Series(sky_model_for_day(date).angles.flatten())

angles = angles.T

mds = manifold.MDS(n_components=2)
X_projected = mds.fit_transform(angles)

plt.figure()
for index, (x, y) in enumerate(X_projected):
    name = angles.index[index].strftime('%d.%m %Hh')
    plt.scatter([x], [y])
    plt.text(x, y, name, color=(0,0,0), fontdict={'weight': 'bold', 'size': 10})

plt.show()
