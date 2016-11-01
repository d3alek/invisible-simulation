from pysolar.solar import *
from pysolar.util import get_sunrise_sunset
import datetime
import numpy as np
import pytz

LATITUDE_SEVILLA_LONGITUDE_SEVILLA = (37.366123, -5.996422)

LATITUDE_LONGITUDE_EDINBURGH = (55.9449353, -3.1839465)

LATITUDE, LONGITUDE = LATITUDE_LONGITUDE_EDINBURGH
TIMEZONE = 'Europe/London'

def sunrise_sunset(when):
    return get_sunrise_sunset(LATITUDE, LONGITUDE, when)

def pysolar_to_local(pysolar_position):
    altitude, azimuth = pysolar_position
    return np.deg2rad(altitude), np.deg2rad((-azimuth + 180) % 360)

def sun_position(datetime_utc):
    # The conversion from datetime_utc to raw_localized_datetime is necessary because
    # get_altitude and get_azimuth assume the time is given in local time, but not
    # the correct tzinfo way, that's why we add the DST value to the UTC time.
    local_datetime = pytz.timezone(TIMEZONE).localize(datetime_utc)
    raw_localized_datetime = datetime_utc + local_datetime.dst()

    return pysolar_to_local((get_altitude(LATITUDE, LONGITUDE, raw_localized_datetime), get_azimuth(LATITUDE, LONGITUDE, raw_localized_datetime)))

