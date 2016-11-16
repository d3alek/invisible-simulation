from operator import itemgetter
import csv
import sys
from matplotlib import pyplot as plt
import datetime

def plot_data(file_name):
    data = [*csv.reader(open(file_name), delimiter=" ")]
    data = [*map(lambda row: (datetime.datetime.strptime(row[0], '%Y-%m-%d'), float(row[1]), float(row[2])), data)]

    x = [*map(itemgetter(0), data)]
    means = [*map(itemgetter(1), data)]
    medians = [*map(itemgetter(2), data)]
    plt.plot(x, means, label="means")
    plt.plot(x, medians, label="medians")

plot_data(sys.argv[1])
if len(sys.argv) > 2:
    plot_data(sys.argv[2])

plt.legend()
plt.show()

