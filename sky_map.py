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
yellowColor = pygame.Color(255, 255, 0)

arrow=pygame.Surface((20,20)) # square so that the engine does not cut the image due to rounding
arrow.fill(blackColor)
pygame.draw.line(arrow, whiteColor, (15,0), (20,5))
pygame.draw.line(arrow, whiteColor, (15,10), (20,5))
pygame.draw.line(arrow, whiteColor, (0,5), (20,5))
arrow.set_colorkey(blackColor)

sun_at = EAST
ten_degrees_in_radians = np.pi/18
observed_azimuths = np.arange(2*np.pi, step=ten_degrees_in_radians)
observed_radii = np.arange(1, step=0.1) * RADIUS

def radius_to_altitude(radius):
    return (radius/RADIUS)*np.pi/2

def altitude_to_radius(altitude):
    return (altitude/(np.pi/2))*RADIUS

observed_altitudes = np.array([*map(radius_to_altitude, observed_radii)])

def sky_model_cartesian_to_sky_map_cartesian(sky_model_cartesian):
    return sky_model_cartesian[:2][::-1]

def cartesian2d(polar):
    cartesian = sky_model_cartesian_to_sky_map_cartesian(to_cartesian(polar))

    return np.int32(CENTER + cartesian * np.array([RADIUS, RADIUS]))

def draw_sun():
    pygame.draw.circle(windowSurfaceObj, yellowColor, cartesian2d(sun_at), 20, 0)

def drawArrow(angle_rad, pos):
    rotated_arrow = pygame.transform.rotate(arrow, np.rad2deg(angle_rad+np.pi/2)) # adding np.pi/2 empirically determined when comparing to 3D plot (from reproduce_wiki_plots.py)
    rect = rotated_arrow.get_rect(center=pos)
    windowSurfaceObj.blit(rotated_arrow, rect)

def draw_angles():
    sky_model = SkyModelGenerator().with_sun_at(sun_at).generate(observed_altitudes, observed_azimuths)
    for index_altitude, altitude in enumerate(observed_altitudes):
        for index_azimuth, azimuth in enumerate(observed_azimuths):
            pos = cartesian2d((altitude, azimuth))
            drawArrow(sky_model.angles[index_altitude, index_azimuth], pos)

while True:
    windowSurfaceObj.fill(blackColor)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.event.post(pygame.event.Event(QUIT))

    draw_angles()

    draw_sun()
    pygame.display.update()
    fpsClock.tick(10)


