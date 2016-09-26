import pygame, sys
from features.sky_model import SkyModel
import numpy as np
from pygame.locals import *

pygame.init()
fpsClock = pygame.time.Clock()

windowSurfaceObj = pygame.display.set_mode((640, 480))
pygame.display.set_caption('Sky Map')

blackColor = pygame.Color(0, 0, 0)

while True:
    windowSurfaceObj.fill(blackColor)

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.event.post(pygame.event.Event(QUIT))


    pygame.display.update()
    fpsClock.tick(30)
