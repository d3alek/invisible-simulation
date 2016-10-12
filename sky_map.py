"""
Observer is facing north and looking up straight at the zenith. This means that:
    - north is up, east is right
    - sun rises right and sets left

"""

import pygame, sys
from features.sky_model import SkyModelGenerator, to_cartesian
import numpy as np
from pygame.locals import *
import ipdb

from pysolar.solar import *
import datetime

pygame.init()
fpsClock = pygame.time.Clock()

EAST = (0, np.pi/2)

RADIUS = 400
LATITUDE_SEVILLA = 37.366123
LONGITUDE_SEVILLA = -5.996422

WIDTH = 2*RADIUS + 100
HEIGHT = WIDTH

CENTER = np.array([WIDTH, HEIGHT])/2

windowSurfaceObj = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Sky Map')

blackColor = pygame.Color(0, 0, 0)
whiteColor = pygame.Color(255, 255, 255)
redColor = pygame.Color(255, 0, 0)
yellowColor = pygame.Color(255, 255, 0)

sun_at = EAST
ten_degrees_in_radians = np.pi/18
observed_azimuths = np.arange(2*np.pi, step=ten_degrees_in_radians)
observed_radii = np.arange(1, step=0.1) * RADIUS

def radius_to_altitude(radius):
    return (radius/RADIUS)*np.pi/2

def altitude_to_radius(altitude):
    return (altitude/(np.pi/2))*RADIUS

observed_altitudes = np.array([*map(radius_to_altitude, observed_radii)])

def cartesian_to_polar(cartesian):
    x, y = cartesian
    r = np.sqrt(x*x + y*y)
    azimuth = np.arctan2(y, x)

    return -r * np.pi/2 + np.pi/2, azimuth % (2*np.pi)

def polar(sky_map_coordinates):
    """Opposite of cartesian2d. For some reason this doctest does not execute

    >>> p = (1.5, 4.7)
    >>> print(np.round((polar(cartesian2d(p))), 1))
    [ 1.5  4.7]
    >>> p = (0.5, 0.3)
    >>> print(np.round((polar(cartesian2d(p))), 1))
    [ 0.5  0.3]
    """
   
    non_centered = (sky_map_coordinates - CENTER) / np.array([RADIUS, RADIUS])
    x, y = non_centered
    sky_model_cartesian = -y, x

    return cartesian_to_polar(sky_model_cartesian)

def sky_model_cartesian_to_sky_map_cartesian(sky_model_cartesian):
    y, x = sky_model_cartesian[:2]

    return x, -y # need to invert y otherwise the azimuth rotates the wrong way (visible from the sun rotation)

def cartesian2d(polar):
    cartesian = sky_model_cartesian_to_sky_map_cartesian(to_cartesian(polar))

    return np.int32(CENTER + cartesian * np.array([RADIUS, RADIUS]))

def draw_sun():
    pygame.draw.circle(windowSurfaceObj, yellowColor, cartesian2d(sun_at), 20, 0)

def draw_arrow(angle_rad, pos, width=1):
    arrow=pygame.Surface((20,20)) # square so that the engine does not cut the image due to rounding
    arrow.fill(blackColor)
    pygame.draw.line(arrow, whiteColor, (15,0), (20,5), width)
    pygame.draw.line(arrow, whiteColor, (15,10), (20,5), width)
    pygame.draw.line(arrow, whiteColor, (0,5), (20,5), width)
    arrow.set_colorkey(blackColor)

    # adding np.pi/2 empirically determined when comparing to 3D plot (from reproduce_wiki_plots.py)
    rotated_arrow = pygame.transform.rotate(arrow, np.rad2deg(angle_rad)) 
    rect = rotated_arrow.get_rect(center=pos)
    windowSurfaceObj.blit(rotated_arrow, rect)

    fontObj = pygame.font.Font('freesansbold.ttf', 13)
    renderedFont = fontObj.render(str(np.int8(np.round(np.rad2deg(angle_rad), 0))), False, redColor)
    rect = renderedFont.get_rect(center=pos+5)
    windowSurfaceObj.blit(renderedFont, rect)

def draw_angles():
    sky_model = SkyModelGenerator().with_sun_at(sun_at).generate(observed_altitudes, observed_azimuths)
    for index_altitude, altitude in enumerate(observed_altitudes):
        for index_azimuth, azimuth in enumerate(observed_azimuths):
            pos = cartesian2d((altitude, azimuth))
            angle = sky_model.angles[index_altitude, index_azimuth]
            degree = sky_model.degrees[index_altitude, index_azimuth]
            draw_arrow(angle, pos, int(1+5*degree))

date = datetime.date.today()
def calculate_sunrise_sunset_times():
    times = datetime.time(8, 0), datetime.time(18, 0)
    return [*map(lambda a: datetime.datetime.combine(date, a), times)]

def pysolar_to_local(pysolar_position):
    altitude, azimuth = pysolar_position
    return np.deg2rad(altitude), np.deg2rad((-azimuth + 180) % 360)

def sun_position(datetime):
    return pysolar_to_local((get_altitude(LATITUDE_SEVILLA, LONGITUDE_SEVILLA, datetime), get_azimuth(37.366123, -5.996422, datetime)))

sunrise_time, sunset_time = calculate_sunrise_sunset_times()
day_length = sunset_time - sunrise_time
hours = day_length.seconds/(3600)
minutes = (day_length.seconds%(3600))/60
print("Day length is %d:%02d hours" % (hours, minutes))

sun_at = sun_position(sunrise_time)
mouse_down = False

date = datetime.date.today()
def calculate_sunrise_sunset_times():
    times = datetime.time(8, 0), datetime.time(18, 0)
    return [*map(lambda a: datetime.datetime.combine(date, a), times)]

def pysolar_to_local(pysolar_position):
    altitude, azimuth = pysolar_position
    return np.deg2rad(altitude), np.deg2rad((-azimuth + 180) % 360)

def sun_position(datetime):
    return pysolar_to_local((get_altitude(LATITUDE_SEVILLA, LONGITUDE_SEVILLA, datetime), get_azimuth(LATITUDE_SEVILLA, LONGITUDE_SEVILLA, datetime)))

sunrise_time, sunset_time = calculate_sunrise_sunset_times()
day_length = sunset_time - sunrise_time
hours = day_length.seconds/(3600)
minutes = (day_length.seconds%(3600))/60
print("Day length is %d:%02d hours" % (hours, minutes))

sun_at = sun_position(sunrise_time)
mouse_down = False

def print_angle_and_degree_at(sky_map_coordinates):
    observed = polar(sky_map_coordinates)
    sky_model_generator = SkyModelGenerator().with_sun_at(sun_at)
    print("Observed: %s Angle: %f Degree %f" % (np.rad2deg(observed), np.rad2deg(sky_model_generator.get_angle(observed)), sky_model_generator.get_degree(observed)))

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    while True:
        windowSurfaceObj.fill(blackColor)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                mouse_down = True
                moved = False
            elif event.type == MOUSEBUTTONUP:
                mouse_down = False
                if not moved:
                    print_angle_and_degree_at(event.pos)
            elif event.type == MOUSEMOTION:
                mousex, mousey = event.pos
                if mouse_down:
                    moved = True
                    ratio = 1 - mousex/WIDTH
                    sun_at = sun_position(sunrise_time + day_length * ratio)

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))

        draw_angles()

        draw_sun()
        pygame.display.update()
        fpsClock.tick(10)

