from sklearn import manifold
import datetime
import sun_calculator as sc
from features.sky_model import SkyModelGenerator as smg
import matplotlib.pyplot as plt
import numpy as np

def sky_model_for_day(day):
        return smg(sc.sun_position(day), yaw=0).generate()

day = datetime.datetime.strptime('160801 10:00', '%y%m%d %H:%M')

month_angles = []
for n in range(30):
    month_angles.append(sky_model_for_day(day + datetime.timedelta(hours=n)).angles.flatten())

mds = manifold.MDS(n_components=2)
X_projected = mds.fit_transform(month_angles)

plt.scatter(X_projected[:,0], X_projected[:,1], color=plt.cm.cubehelix(np.linspace(0,1,n)))
plt.show()
