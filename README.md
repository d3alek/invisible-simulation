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
                        use feature ranking (comes out of another script). 0
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

