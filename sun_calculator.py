from pysolar.solar import *
from pysolar.util import get_sunrise_sunset
import datetime
import numpy as np
from geometry import PolarPoint
from places import sevilla


def sunrise_sunset(time, place = sevilla):
    return get_sunrise_sunset(place.latitude, place.longitude, time)

def pysolar_to_local(pysolar_position):
    return PolarPoint(pysolar_position.altitude, (-pysolar_position.azimuth + np.pi) % (2*np.pi))

def sun_position(time_utc, place = sevilla):
    # The conversion from datetime_utc to raw_localized_datetime is necessary because
    # get_altitude and get_azimuth assume the time is given in local time, but not
    # the correct tzinfo way, that's why we add the DST value to the UTC time.
    local_datetime = place.timezone.localize(time_utc)
    raw_localized_datetime = time_utc + local_datetime.dst()

    altitude = get_altitude(place.latitude, place.longitude, raw_localized_datetime)
    azimuth = get_azimuth(place.latitude, place.longitude, raw_localized_datetime)
    return pysolar_to_local(PolarPoint(np.deg2rad(altitude), np.deg2rad(azimuth)))

