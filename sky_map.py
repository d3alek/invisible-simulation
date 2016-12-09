"""
Observer is facing north and looking up straight at the zenith. This means that:
    - north is up, east is right
    - sun rises right and sets left

"""

import pygame, sys
from sky_model import SkyModelGenerator
from geometry import PolarPoint
import numpy as np
from pygame.locals import *
import ipdb

import datetime
from sun_calculator import sunrise_sunset, sun_position

import viewers

import matplotlib.pyplot as plt

pygame.init()
fpsClock = pygame.time.Clock()

FONT_TTF = 'freesansbold.ttf'
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
        fontObj = pygame.font.Font(FONT_TTF, 13)
        text = str(np.int16(np.round(np.rad2deg(angle_rad), 0)))
        renderedFont = fontObj.render(text, False, redColor)
        rect = renderedFont.get_rect(center=pos+5)
        windowSurfaceObj.blit(renderedFont, rect)

def draw_north(yaw_degrees):
    arrow = draw_arrow(redColor, 3, 90+yaw_degrees)
    position = np.array([WIDTH - 100, 100])
    rect = arrow.get_rect(center=position)
    windowSurfaceObj.blit(arrow, rect)
    font = pygame.font.Font(FONT_TTF, 13)
    rendered_font = font.render("N", False, redColor)
    rotated_rendered_font = pygame.transform.rotate(rendered_font, yaw_degrees)
    font_position = np.array([np.sin(np.deg2rad(yaw_degrees))*20,np.cos(np.deg2rad(yaw_degrees))*20])
    rect = rotated_rendered_font.get_rect(center=position - font_position)
    windowSurfaceObj.blit(rotated_rendered_font, rect)

def draw_angles(sky_model_generator):
    sky_model = sky_model_generator.generate(viewer=viewers.uniform_viewer())#viewers.vertical_strip_viewer())
    for index, (altitude, azimuth) in enumerate(sky_model.observed_points):
        angle = sky_model.angles[index]
        degree = sky_model.degrees[index]
        draw_angle_arrow(angle, (altitude, azimuth), sky_model.yaw, int(1+5*degree))

def draw_intensity(sky_model_generator):
    sky_model = sky_model_generator.generate(viewer=viewers.uniform_viewer())#viewers.vertical_strip_viewer())
    intensities = sky_model.intensities
    white = np.array([[255,255,255]]*len(intensities)) # TODO do it black white
    #colors = , normalize(intensities))
    for (altitude, azimuth), color in zip(sky_model.observed_points, colors):
        pygame.draw.circle(windowSurfaceObj, 255*color, cartesian2d((altitude, azimuth)), 10, 0)

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
    sky_model_local_observed = PolarPoint.from_tuple(observed)
    print("Observed: %s Angle: %f Degree %f" % ([*map(np.rad2deg, sky_model_local_observed)], np.rad2deg(sky_model_generator.get_angle(sky_model_local_observed)), sky_model_generator.get_degree(sky_model_local_observed)))

def normalize(array):
    return ((-array.min() + array) * (255/(-array.min() + array.max()))).astype(int)

def draw_predictors(polar_ranks, rank_at_most):
    normalized = normalize(polar_ranks[:,1].astype(int))
    m = normalized.max()
    normalized[polar_ranks[:,1] > rank_at_most] = m
    colors = plt.cm.cubehelix(normalized)
    for polar, parameter, color in zip(polar_ranks[:,0], polar_ranks[:,1], colors):
        pygame.draw.circle(windowSurfaceObj, 255*(1-color), cartesian2d(polar), 10, 0)

def add_polar_coordinates(ranks):
    """ Makes the implicit assumption that the used viewer was uniform viewer """
    l = len(ranks)
    angle_sin_ranks = ranks[:l/3]
    angle_cos_ranks = ranks[l/3:2*l/3]
    degree_ranks = ranks[2*l/3:]
    assert len(angle_sin_ranks) == len(angle_cos_ranks) and len(degree_ranks) == len(angle_cos_ranks)

    polar_coordinates = viewers.uniform_viewer().get_observed_points()

    return {"degree": np.array([*zip(polar_coordinates, degree_ranks)]), 
            "angle-sin": np.array([*zip(polar_coordinates, angle_sin_ranks)]), 
            "angle-cos": np.array([*zip(polar_coordinates, angle_cos_ranks)])}

def print_statusbar(string):
    fontObj = pygame.font.Font(FONT_TTF, 20)
    renderedFont = fontObj.render(string, False, whiteColor)
    rect = renderedFont.get_rect(center=(WIDTH/2, HEIGHT-10))
    windowSurfaceObj.blit(renderedFont, rect)

def print_help():
    print("An interactive visualization of the polarized sky")
    print("Drag mouse right and left to move the sun from sunrise to sunset and back")
    print("Hit escape to exit")
    print("Other commands:")
    print(" A click at a point tells you detailed information about the parameters of the sky at that point")
    print(" i shows the intensity distribution of the sky")
    print(" left and right keyboard keys simulate the agent rotating on the spot. Observe the red arrow that always points north.")
    print(" p turns predictor display on, showing how useful the sampled points of the sky are. Brighter is higher")
    print("   d shows degree of polarization features ranking")
    print("   s shows polarization angle sin features ranking")
    print("   a shows polarizaiton angle cos features ranking")
    print("   c uses ranking for yaw cos prediction")
    print("   v uses ranking for yaw sin prediction")
    print("   keys 1-4 show the top [1, 20, 100, 1000] features")
if __name__ == "__main__":
    print_help()

    import doctest
    doctest.testmod()

    show_predictors = False
    show_degree_predictors = False

    yaw = 0

    rank_at_most = 1000
    show_intensity = False

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
                if event.key == K_s:
                    show_predictors_key = "angle-sin"
                if event.key == K_a:
                    show_predictors_key = "angle-cos"

                if event.key == K_1:
                    rank_at_most = 1

                if event.key == K_2:
                    rank_at_most = 20

                if event.key == K_3:
                    rank_at_most = 100

                if event.key == K_4:
                    rank_at_most = 1000

                if event.key == K_i:
                    show_intensity = not show_intensity

                if event.key == K_LEFT:
                    yaw += 10
                if event.key == K_RIGHT:
                    yaw -= 10

        sky_model_generator = SkyModelGenerator(sun_at, yaw=np.deg2rad(yaw))

        if show_intensity:
            draw_intensity(sky_model_generator)

        else:
            draw_angles(sky_model_generator)

            if show_predictors:
                draw_predictors(features_rank[show_predictors_key], rank_at_most)
                print_statusbar(" ".join([str(rank_at_most), features_rank_file, show_predictors_key]))

        draw_sun(sky_model_generator, sun_at)
        draw_north(yaw)

        pygame.display.update()
        fpsClock.tick(10)


