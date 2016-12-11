# Requirements
- python3
- numpy
- pysolar (for sun position calculation)
- pandas (for sky generator and yaw predictor)
- sklearn (for yaw predictors)
- scipy (for yaw predictor)
- pygame (for sky_map)
- matplotlib (for visualization)
- behave (for running behavioural tests)
- ipdb (for interractive debugging, only used if the behavioural tests go bad)

# Verification of the model

$ python geometry.py
... no output means all doctests passed ...```
$ python sky_model.py 
Sunset looks like sunrise

$ behave
... you expect no failed tests ...
$ python sky_dome_3d.py
... three pyplot skydome plots with the angle of polarization shown as lines
and the degree of polarization shown as the width of the lines ...
$ python reproduce_wiki_graphs.py
... reproduces 2 graphs from https://en.wikipedia.org/wiki/Rayleigh_sky_model to verify visually that the model is correct ...

```

# Data generation
```
usage: sky_generator.py [-h] [--days N] [--start START] [--end END]
                        [--freq FREQ] [--hours HOURS] [--yaw-step YAW_STEP]
                        [--place PLACE]
                        date

Do a linear regression on a sample of the sky over N days to predict the time
of day.

positional arguments:
  date                 start date for training data generation. Format is
                       160501 for May 1 2016

optional arguments:
  -h, --help           show this help message and exit
  --days N             the number of days to gather training data for. Default
                       1
  --start START        the number of days to gather training data for. Default
                       6
  --end END            the number of days to gather training data for. Default
                       20
  --freq FREQ          Mutually exclusive with --hours. Sample the sky between
                       *start* hour and *end* hour at frequency - 10min, 1H,
                       1D
  --hours HOURS        Mutually exclusive with freq. Sample the sky at this
                       hour each day (can be specified multiple times)
  --yaw-step YAW_STEP  rotational step in degrees. Default 10 degrees.
  --place PLACE        place on Earth to simulate the sky for. To add a place,
                       edit places.py. Default sevilla. Available places:
                       ['edinburgh', 'sevilla'].
```
```
$ python sky_generator.py 160501 --days=1 --freq=10min --yaw-step=10
Generating skies for 85 datetimes starting at 2016-05-01 06:00:00 and ending at 2016-05-01 20:00:00 at 10 degree step rotations
3060 skies in total
...
Stored data as csv: skies/160501-sevilla-1-10min-10.csv

$ python sky_generator.py 160501 --days=1 --hours=8 --hours=10 --hours=12 --hours=14 --hours=16 --hours=18 --yaw-step=10
( equivalent result can be acheived by doing python sky_generator.py 160501 --days=1 --start=8 --end=18 --freq=2H --yaw-step=10 )
...
Stored data as csv: skies/160501-sevilla-1-8-10-12-14-16-18-10.csv
```

# Prediction
```
$ python yaw_predictor.py -h
usage: yaw_predictor.py [-h] [--polar] [--use-time]
                        [--lowest-rank LOWEST_RANK]
                        training test

Train a linear regression on the sky and evaluate the resulting model.

positional arguments:
  training              training dataset csv file path
  test                  test dataset csv file path

optional arguments:
  -h, --help            show this help message and exit
  --polar               produce polar plot
  --use-time            use time as a feature
  --lowest-rank LOWEST_RANK
                        use feature ranking (comes out of yaw_feature_selection.py). 0
                        disables it (default), 1 means use only features
                        ranked 1, etc.
```

```
$ python yaw_predictor.py skies/160501-sevilla-1-10min-10.csv skies/160501-sevilla-1-8-10-12-14-16-18-10.csv --polar
Ridge(alpha=1000, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001)
Saved figure to graphs/Ridge_on_training[160501-sevilla-1-10min-10]_test[160501-sevilla-1-8-10-12-14-16-18-10]_polar_972_features.png
```
![Ridge_on_training[160501-sevilla-1-10min-10]_test[160501-sevilla-1-8-10-12-14-16-18-10]_polar_972_features.png](graphs/example.png)

# Feature selection
Passing the --parallel argument is recommended because computation takes a lot of time. However, it is known to crash the feature selection on Bena's macbook.
```
$ python yaw_feature_selection.py skies/160501-sevilla-1-10min-10.csv
... wait for  hours ...
Saved pickled rfe_sin and rfe_cos in data directory
```

Two rankings are produced because the yaw is divided in sin and cos to model its cycliclty. 
Read the pickled objects's ranking_ parameter to see for a list of ranks where 1 is best rank.

```
$ python yaw_predictor.py skies/160501-sevilla-1-10min-10.csv skies/160501-sevilla-1-8-10-12-14-16-18-10.csv --polar --lowest-rank=1
... should produce worse predictions using only the features ranked 1 ...

$ python sky_map
```

Press P to see degree ranking with colors (lighter is lower rank, better ranked).
Press S to see ranking of the sin component of the angle features.
Press C to see the yaw cos predictor ranking.
Press 1 to see the features ranked 1, 2 to see the features with rank <=20 and so on.
```
