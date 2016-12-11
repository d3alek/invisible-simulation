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

import datetime
from sun_calculator import sunrise_sunset, sun_position

import viewers

import matplotlib.pyplot as plt
from os.path import isfile

import pickle

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

STATE_ANGLE_DEGREES = "state-angle-degrees"
STATE_FEATURE_RANKS = "state-feature-ranks"
STATE_INTENSITY = "state-intensity"
states = [STATE_ANGLE_DEGREES, STATE_FEATURE_RANKS, STATE_INTENSITY]

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
    sky_model = sky_model_generator.generate(viewer=viewers.uniform_viewer())
    for index, (altitude, azimuth) in enumerate(sky_model.observed_points):
        angle = sky_model.angles[index]
        degree = sky_model.degrees[index]
        draw_angle_arrow(angle, (altitude, azimuth), sky_model.yaw, int(1+5*degree))

def normalize(array):
    return ((-array.min() + array) * (255/(-array.min() + array.max()))).astype(int)

def draw_intensity(sky_model_generator):
    sky_model = sky_model_generator.generate(viewer=viewers.uniform_viewer())
    intensities = sky_model.intensities
    normalized = normalize(intensities)
    colors = plt.cm.gray(normalized)

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
    smg = SkyModelGenerator(sun_at, yaw=yaw)
    observed_point = PolarPoint.from_tuple(observed)
    print("Observed: %s Angle: %f Degree %f Intensity: %f" % ([*map(np.rad2deg, observed_point)], np.rad2deg(smg.get_angle(observed_point)), smg.get_degree(observed_point), smg.get_intensity(observed_point)))

def draw_feature_ranks(polar_ranks):
    if polar_ranks.size == 0:
        return

    normalized = normalize(polar_ranks[:,1].astype(int))
    colors = plt.cm.gray(normalized)
    for polar, parameter, color in zip(polar_ranks[:,0], polar_ranks[:,1], colors):
        pygame.draw.circle(windowSurfaceObj, 255*(1-color), cartesian2d(polar), 10, 0)

def split_into_feature_types(ranks):
    l = len(ranks)
    angle_sin_ranks = ranks[:l/3]
    angle_cos_ranks = ranks[l/3:2*l/3]
    degree_ranks = ranks[2*l/3:]
    assert len(angle_sin_ranks) == len(angle_cos_ranks) and len(degree_ranks) == len(angle_cos_ranks)

    return {"degree": degree_ranks,
            "angle-sin": angle_sin_ranks, 
            "angle-cos": angle_cos_ranks}

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


    yaw = 0

    rank_at_most = 1000
    state = states[0]
    feature_rank_files = {
            "sin" : "data/rfe_sin.pickle",
            "cos": "data/rfe_cos.pickle"
            }

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
                    if state == state[0]:
                        pygame.event.post(pygame.event.Event(QUIT))
                    else:
                        state = state[0]

                elif event.key == K_p:
                    if isfile(feature_rank_files['sin']) and isfile(feature_rank_files['cos']):
                        state = STATE_FEATURE_RANKS
                        ranking_yaw_component = 'sin'
                        ranking_feature_type = "angle-sin"
                    else:
                        print("At least one of %s does not exist. Please run yaw_feature_predictor.py to generate the files" % feature_rank_files.values())

                elif state == STATE_FEATURE_RANKS:
                    if event.key == K_c:
                        ranking_yaw_component = 'cos'
                    elif event.key == K_v:
                        ranking_yaw_component = 'sin'
                    elif event.key == K_d:
                        ranking_feature_type = "degree"
                    elif event.key == K_s:
                        ranking_feature_type = "angle-sin"
                    elif event.key == K_a:
                        ranking_feature_type = "angle-cos"

                    elif event.key == K_1:
                        rank_at_most = 1
                    elif event.key == K_2:
                        rank_at_most = 20
                    elif event.key == K_3:
                        rank_at_most = 100
                    elif event.key == K_4:
                        rank_at_most = 1000

                if event.key == K_i:
                    state = STATE_INTENSITY

                if event.key == K_LEFT:
                    yaw += 10
                if event.key == K_RIGHT:
                    yaw -= 10

        sky_model_generator = SkyModelGenerator(sun_at, yaw=np.deg2rad(yaw))

        if state == STATE_INTENSITY:
            draw_intensity(sky_model_generator)

        else:
            draw_angles(sky_model_generator)

            if state == STATE_FEATURE_RANKS:
                ranking = pickle.load(open(feature_rank_files[ranking_yaw_component], 'rb')).ranking_

                ranked_feature_types = split_into_feature_types(ranking)

                ranked = ranked_feature_types[ranking_feature_type]
                observed_points = np.array(viewers.uniform_viewer().get_observed_points())
                assert len(ranked) == len(observed_points), "Ranking has different number of features that the uniform viewer, expected %d got %d" % (len(observed_points), len(ranked))

                ranked_trimmed = ranked[ranked <= rank_at_most]
                observed_points_trimmed = observed_points[ranked <= rank_at_most]

                rank_coordinates = np.array([*zip(observed_points_trimmed, ranked_trimmed)])

                draw_feature_ranks(rank_coordinates)
                print_statusbar("Showing top %d %s features that predict yaw %s component" % (rank_at_most, ranking_feature_type, ranking_yaw_component))

        draw_sun(sky_model_generator, sun_at)
        draw_north(yaw)

        pygame.display.update()
        fpsClock.tick(10)


