This readme serves as a specification of the desired behavior of the scripts in this repository. 

What you will notice is I like the question asking approach instead of exploding and redirecting to help if a required parameter is missing.
This has the advantage that you don't forget about some optional parameters. If you want the optional parameters to be set to default, pass in --defaults.

# Model automated verification

```
$ python features/sky_model.py
$ behave
```

# Data generation
```
$ python sky_generator.py
From date? (--date=160501): 
Days? (--days=1): 
Frequency? (--freq=10min): 
Yaw step? (--yaw-step=10): # At each time point, yaw from 0 to 360 at yaw-step steps
Location (edinburgh, sevilla or long,lat)? (--location=sevilla): 
Keep yaw and angle of polarization as a scalar instead the default decomposing into sin cos, yes/NO? (--scalar-yaw): 
Include time, millis/binary/NO? (--time=no): # binary means 0 for morning 1 for afternoon, default is don't use time
Label? (--label=string):

Executing command:
python sky_generator.py --date=...

Generating skies for 85 datetimes starting at 2016-05-01 06:00:00 and ending at 2016-05-01 20:00:00 at 10 degree step rotations
3060 skies in total

...

Stored data as csv: skies/160501-1-10min-10.csv

Done!

```

# Prediction

```
$ python yaw_predictor.py
Training file (generated from sky_generator.py)? (--training=FILE): 
Test file (generated from sky_generator.py)? (--test=FILE): 
Regressor? (--regressor=linear): 
Generate linear plots instead of polar, yes/NO? (--linear):

Executing command:
python yaw-predictor.py ...

Generated files:
...

Done!

```

