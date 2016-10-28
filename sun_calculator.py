from pysolar.solar import *
from pysolar.util import get_sunrise_sunset
import datetime
import numpy as np

LATITUDE_SEVILLA_LONGITUDE_SEVILLA = (37.366123, -5.996422)

LATITUDE_LONGITUDE_EDINBURGH = (55.9449353, -3.1839465)

LATITUDE, LONGITUDE = LATITUDE_LONGITUDE_EDINBURGH

def sunrise_sunset(when):
    return get_sunrise_sunset(LATITUDE, LONGITUDE, when)

def pysolar_to_local(pysolar_position):
    altitude, azimuth = pysolar_position
    return np.deg2rad(altitude), np.deg2rad((-azimuth + 180) % 360)

def sun_position(datetime):
    return pysolar_to_local((get_altitude(LATITUDE, LONGITUDE, datetime), get_azimuth(LATITUDE, LONGITUDE, datetime)))

