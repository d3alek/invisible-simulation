from geometry import PolarPoint
import pytz

class Place:
    def __init__(self, latitude, longitude, timezone):
        """
        Latitude and logitude of a place in degrees. You can use google maps's url parameter @x,y
        Timezone as a string that pytz can understand, for example Europe/London
        """
        self.latitude, self.longitude = latitude, longitude
        self.timezone = pytz.timezone(timezone)

sevilla = Place(37.366123, -5.996422, 'Europe/Madrid')
edinburgh = Place(55.9449353, -3.1839465, 'Europe/London')
