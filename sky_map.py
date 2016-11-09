"""
Observer is facing north and looking up straight at the zenith. This means that:
    - north is up, east is right
    - sun rises right and sets left

"""

import pygame, sys
from features.sky_model import SkyModelGenerator, LocalPolar
import numpy as np
from pygame.locals import *
import ipdb

import datetime
from features.sun_calculator import sunrise_sunset, sun_position

import features.viewers as viewers

pygame.init()
fpsClock = pygame.time.Clock()

EAST = (0, np.pi/2)

RADIUS = 400

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

def polar(sky_map_coordinates):
    """Opposite of cartesian2d. For some reason this doctest does not execute

    >>> p = (1.5, 4.7)
    >>> print(np.round((polar(cartesian2d(p))), 1))
    [ 1.5  4.7]
    >>> p = (0.5, 0.3)
    >>> print(np.round((polar(cartesian2d(p))), 1))
    [ 0.5  0.3]
    """

    distance_to_center_vector = sky_map_coordinates - CENTER
    r = np.linalg.norm(distance_to_center_vector) / RADIUS
    azimuth = np.arctan2(distance_to_center_vector[0], -distance_to_center_vector[1])

    return (1-r)*np.pi/2, azimuth % (2*np.pi)


def sky_model_cartesian_to_sky_map_cartesian(sky_model_cartesian):
    x, y = sky_model_cartesian[:2]

    return y, -x # need to invert y otherwise the azimuth rotates the wrong way (visible from the sun rotation)

def cartesian2d(polar):
    altitude, azimuth = polar
    r = 1 - altitude/(np.pi/2)
    x = np.sin(azimuth)*r 
    y = -np.cos(azimuth)*r

    cartesian = x, y

    return np.int32(CENTER + cartesian * np.array([RADIUS, RADIUS]))

def draw_sun(sky_model_generator, sun_position):
    pygame.draw.circle(windowSurfaceObj, yellowColor, cartesian2d(sky_model_generator.sun), 20, 0)

def draw_arrow(color, width, rotated=0):
    arrow=pygame.Surface((20,20)) # square so that the engine does not cut the image due to rounding
    arrow.fill(blackColor)
    pygame.draw.line(arrow, color, (15,0), (20,5), width)
    pygame.draw.line(arrow, color, (15,10), (20,5), width)
    pygame.draw.line(arrow, color, (0,5), (20,5), width)
    arrow.set_colorkey(blackColor)

    return pygame.transform.rotate(arrow, rotated) 

def draw_angle_arrow(angle_rad, radians, yaw_radians, width=1, with_text=False):
    azimuth = radians[1]
    pos = cartesian2d(radians)

    # necessary because each point's angle is calculated as if observer is looking towards that point
    # but in the visualization observer is assumed to be looking at yaw
    rotate = np.rad2deg(angle_rad - azimuth)
    arrow = draw_arrow(whiteColor, width, rotated=rotate)

    rect = arrow.get_rect(center=pos)
    windowSurfaceObj.blit(arrow, rect)

    if with_text:
        fontObj = pygame.font.Font('freesansbold.ttf', 13)
        text = str(np.int16(np.round(np.rad2deg(angle_rad), 0)))
        renderedFont = fontObj.render(text, False, redColor)
        rect = renderedFont.get_rect(center=pos+5)
        windowSurfaceObj.blit(renderedFont, rect)

def draw_looking_at(yaw_degrees):
    arrow = draw_arrow(redColor, 3, 90+yaw_degrees)
    rect = arrow.get_rect(center=CENTER)
    windowSurfaceObj.blit(arrow, rect)

def draw_angles(sky_model_generator):
    sky_model = sky_model_generator.generate(observed_polar=viewers.vertical_strip_viewer())
    for index, (altitude, azimuth) in enumerate(sky_model.observed_polar):
        angle = sky_model.angles[index]
        degree = sky_model.degrees[index]
        draw_angle_arrow(angle, (altitude, azimuth), sky_model.yaw, int(1+5*degree))

if len(sys.argv) > 1:
    when = datetime.datetime.strptime(sys.argv[1], "%y%m%d")
else:
    when = datetime.datetime.utcnow()

sunrise_time, sunset_time = sunrise_sunset(when)
print("Sunrise: %s, Sunset %s" % (sunrise_time, sunset_time))
day_length = sunset_time - sunrise_time
hours = day_length.seconds/(3600)
minutes = (day_length.seconds%(3600))/60
print("Day length is %d:%02d hours" % (hours, minutes))

sun_at = sun_position(sunrise_time)
mouse_down = False

def print_angle_and_degree_at(sky_map_coordinates, yaw):
    observed = polar(sky_map_coordinates)
    sky_model_generator = SkyModelGenerator(sun_at, yaw=yaw)
    sky_model_local_observed = LocalPolar.from_tuple(observed)
    print("Observed: %s Angle: %f Degree %f" % ([*map(np.rad2deg, sky_model_local_observed)], np.rad2deg(sky_model_generator.get_angle(sky_model_local_observed)), sky_model_generator.get_degree(sky_model_local_observed)))

def normalize(array):
    return (array - array.min())/(array.max() - array.min())

def draw_predictors(predictors):
    normalized_parameters = normalize(predictors[1])
    for polar, parameter in zip(predictors[0], normalized_parameters):
        pygame.draw.circle(windowSurfaceObj, redColor, cartesian2d(polar), int(10*parameter), 0)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    show_predictors = False
    show_degree_predictors = False

    yaw = 0

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
                    print_angle_and_degree_at(event.pos, np.deg2rad(yaw))
            elif event.type == MOUSEMOTION:
                mousex, mousey = event.pos
                if mouse_down:
                    moved = True
                    ratio = 1 - mousex/WIDTH
                    sun_at = sun_position(sunrise_time + day_length * ratio)

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))
                if event.key == K_p:
                    show_predictors = not show_predictors
                    if show_predictors:
                        import pickle
                        prediction_result = pickle.load(open('data/polarization_time_predictor_result.pickle', 'rb'))
                if event.key == K_d:
                    show_degree_predictors = not show_degree_predictors
                    print("Show degree predictors: %s" % show_degree_predictors)
                if event.key == K_LEFT:
                    yaw += 10
                    print("Yaw: %d" % yaw)
                if event.key == K_RIGHT:
                    yaw -= 10
                    print("Yaw: %d" % yaw)

        sky_model_generator = SkyModelGenerator(sun_at, yaw=np.deg2rad(yaw))
        draw_angles(sky_model_generator)
        draw_sun(sky_model_generator, sun_at)
        draw_looking_at(yaw)

        if show_predictors:
            if show_degree_predictors:
                draw_predictors(prediction_result['degrees'])
            else:
                draw_predictors(prediction_result['angles'])
        pygame.display.update()
        fpsClock.tick(10)


