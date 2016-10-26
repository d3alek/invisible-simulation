"""
Observer is facing south and looking midway to the zenith. We are seeing from their perspective. This means that:
    - during the day the sun moves left to right, east to west

"""
import pygame, sys 
from features.sky_model import SkyModelGenerator
import numpy as np
from pygame.locals import *
import ipdb

from pysolar.solar import *
import datetime

pygame.init()
fpsClock = pygame.time.Clock()

EAST = (0, np.pi/2)
EARTH_RADIUS = 400

LATITUDE_SEVILLA = 37.366123
LONGITUDE_SEVILLA = -5.996422

WIDTH = 2*EARTH_RADIUS
HEIGHT = EARTH_RADIUS

windowSurfaceObj = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Sky Sample')

sun_at = EAST
observed = (np.pi/4, np.pi) #SOUTH at an altitude of 45 degrees

def sky_model_cartesian_to_sky__cartesian(sky_model_cartesian):
    y, x = sky_model_cartesian[:2]

    return x, -y # need to invert y otherwise the azimuth rotates the wrong way (visible from the sun rotation)

def cartesian2d(polar):
    """
    altitude goes from 0 to pi/2 converts to y going from HEIGHT to 0
    azimuth going from 0 to 2*pi converts to:
      - if between pi/2 and 3*pi/2 maps from 0 to WIDTH
      - otherwise return -1 and print error
    """
    altitude, azimuth = polar
    if azimuth < np.pi/2 or azimuth > 3*np.pi/2:
        print("ERROR: polar coordinates outside of viewbox: %s" % polar)
    y = HEIGHT * (1 - altitude / (np.pi/2))
    x = WIDTH * (azimuth - np.pi/2) / np.pi

    return np.int32(np.array([x, y]))

def draw_sun():
    pygame.draw.circle(windowSurfaceObj, yellowColor, cartesian2d(sun_at), 20, 0)

def draw_arrow(angle_rad, degree, pos):
    width = int(1+5*degree)
    arrow=pygame.Surface((20,20)) # square so that the engine does not cut the image due to rounding
    arrow.fill(blackColor)
    pygame.draw.line(arrow, whiteColor, (15,0), (20,5), width)
    pygame.draw.line(arrow, whiteColor, (15,10), (20,5), width)
    pygame.draw.line(arrow, whiteColor, (0,5), (20,5), width)
    arrow.set_colorkey(blackColor)

    # empirically decided to invert the angle to fix the visualization
    rotated_arrow = pygame.transform.rotate(arrow, np.rad2deg(-angle_rad)) 
    rect = rotated_arrow.get_rect(center=pos)
    windowSurfaceObj.blit(rotated_arrow, rect)

    fontObj = pygame.font.Font('freesansbold.ttf', 13)
    renderedFont = fontObj.render(str(np.int8(np.round(np.rad2deg(angle_rad), 0))), False, redColor)
    rect = renderedFont.get_rect(center=pos+5)
    windowSurfaceObj.blit(renderedFont, rect)

date = datetime.date.today()
def calculate_sunrise_sunset_times():
    times = datetime.time(8, 0), datetime.time(18, 0)
    return [*map(lambda a: datetime.datetime.combine(date, a), times)]

def pysolar_to_local(pysolar_position):
    altitude, azimuth = pysolar_position
    return np.deg2rad(altitude), np.deg2rad((-azimuth + 180) % 360)

def sun_position(datetime):
    return pysolar_to_local((get_altitude(LATITUDE_SEVILLA, LONGITUDE_SEVILLA, datetime), get_azimuth(LATITUDE_SEVILLA, LONGITUDE_SEVILLA, datetime)))

def draw_angle():
    sky_model_generator = SkyModelGenerator().with_sun_at(sun_at)

    altitude, azimuth = observed
    for a in np.arange(0, np.pi/2, step=np.pi/10):
        observed_on_line = (a, azimuth)

        angle = sky_model_generator.get_angle(observed_on_line)
        degree = sky_model_generator.get_degree(observed_on_line)

        draw_arrow(angle, degree, cartesian2d(observed_on_line))

sunrise_time, sunset_time = calculate_sunrise_sunset_times()
day_length = sunset_time - sunrise_time
hours = day_length.seconds/(3600)
minutes = (day_length.seconds%(3600))/60
print("Day length is %d:%02d hours" % (hours, minutes))

sun_at = sun_position(sunrise_time)
mouse_down = False

blackColor = pygame.Color(0, 0, 0)
whiteColor = pygame.Color(255, 255, 255)
redColor = pygame.Color(255, 0, 0)
yellowColor = pygame.Color(255, 255, 0)

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
                ratio = mousex/WIDTH
                sun_at = sun_position(sunrise_time + day_length * ratio)

        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.event.post(pygame.event.Event(QUIT))

    draw_angle()
    draw_sun()
    pygame.display.update()
    fpsClock.tick(10)

