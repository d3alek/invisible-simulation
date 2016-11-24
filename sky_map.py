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

import matplotlib.pyplot as plt

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

    rotate = np.rad2deg(angle_rad)
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
    sky_model = sky_model_generator.generate(observed_polar=viewers.uniform_viewer())#viewers.vertical_strip_viewer())
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
    return (array * (255/array.max())).astype(int)

def draw_predictors(polar_ranks, rank_at_most):
    normalized = normalize(polar_ranks[:,1].astype(int))
    m = normalized.max()
    normalized[normalized > rank_at_most] = m
    colors = plt.cm.cubehelix(normalized)
    for polar, parameter, color in zip(polar_ranks[:,0], polar_ranks[:,1], colors):
        pygame.draw.circle(windowSurfaceObj, 255*(1-color), cartesian2d(polar), 10, 0)

def add_polar_coordinates(ranks):
    l = len(ranks)
    angle_sin_ranks = ranks[:l/3]
    angle_cos_ranks = ranks[l/3:2*l/3]
    degree_ranks = ranks[2*l/3:]
    assert len(angle_sin_ranks) == len(angle_cos_ranks) and len(degree_ranks) == len(angle_cos_ranks)

    polar_coordinates = viewers.uniform_viewer()

    return {"degree": np.array([*zip(polar_coordinates, degree_ranks)]), 
            "angle-sin": np.array([*zip(polar_coordinates, angle_sin_ranks)]), 
            "angle-cos": np.array([*zip(polar_coordinates, angle_cos_ranks)])}

def print_statusbar(string):
    fontObj = pygame.font.Font('freesansbold.ttf', 20)
    renderedFont = fontObj.render(string, False, whiteColor)
    rect = renderedFont.get_rect(center=(WIDTH/2, HEIGHT-10))
    windowSurfaceObj.blit(renderedFont, rect)

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    show_predictors = False
    show_degree_predictors = False

    yaw = 0

    rank_at_most = 1000

    show_predictors_key = "degree"
    features_rank = []

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
                    if not features_rank:
                        import pickle
                        features_rank_file = 'data/rfe_sin.pickle'
                        features_rank = add_polar_coordinates(pickle.load(open(features_rank_file, 'rb')).ranking_)

                if event.key == K_c:
                    import pickle
                    features_rank_file = 'data/rfe_cos.pickle'
                    features_rank = add_polar_coordinates(pickle.load(open(features_rank_file, 'rb')).ranking_)
                if event.key == K_v:
                    import pickle
                    features_rank_file = 'data/rfe_sin.pickle'
                    features_rank = add_polar_coordinates(pickle.load(open(features_rank_file, 'rb')).ranking_)

                if event.key == K_d:
                    show_predictors_key = "degree"
                    print("Show degree predictors")
                if event.key == K_s:
                    show_predictors_key = "angle-sin"
                    print("Show angle sin predictors")
                if event.key == K_a:
                    show_predictors_key = "angle-cos"
                    print("Show angle cos predictors")

                if event.key == K_1:
                    rank_at_most = 1

                if event.key == K_2:
                    rank_at_most = 20

                if event.key == K_3:
                    rank_at_most = 100

                if event.key == K_4:
                    rank_at_most = 1000

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
            draw_predictors(features_rank[show_predictors_key], rank_at_most)
            print_statusbar(" ".join([str(rank_at_most), features_rank_file, show_predictors_key]))

        pygame.display.update()
        fpsClock.tick(10)


