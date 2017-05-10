import numpy as np
import sys
#import random
#import pygame
#import pygame.surfarray as surfarray
#from pygame.locals import *
from itertools import cycle

import os

 
SCREEN_WIDTH=160
SCREEN_HEIGH=320
FPS =60



MAX_CARS_COUNT = 10
CAR_ATTRIBUTE_COUNT =3 # x , y  , speed 
MIN_SPEED = 7/10
MAX_SPEED =  7/10


AGENT_INIT_SPEED = 0/10  
AGENT_MAX_SPEED = 10/10  

TILE_SIZE =16

WORD_WIDTH = 9 * TILE_SIZE 
WORD_HEIGHT = 19 * TILE_SIZE 
"""
pygame.init()
pygame.font.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGH), pygame.RESIZABLE)
pygame.display.set_caption("Simuator")


 
bg_image  = pygame.image.load("data/bg.png").convert_alpha()
car_image = pygame.image.load('data/black.png').convert_alpha()
 

clock = pygame.time.Clock()
"""
starts = [
    {'y': 0 *TILE_SIZE, 'x': 3 *TILE_SIZE, 'start': 'down'},
    {'y': 0*TILE_SIZE , 'x': 4*TILE_SIZE , 'start': 'down'},
    {'y': 19 *TILE_SIZE, 'x': 5*TILE_SIZE , 'start': 'up'},
    {'y': 19 *TILE_SIZE, 'x': 6 *TILE_SIZE, 'start': 'up'}
]  

agent_start = {'y': 10*TILE_SIZE, 'x': -1*TILE_SIZE, 'start': 'right'}

lx1 = 3 *TILE_SIZE
lx2 = 4 *TILE_SIZE
lx3 = 5 *TILE_SIZE
lx4 = 6 *TILE_SIZE

ay= 10*TILE_SIZE 

cars_speed =0.3
 

class GameState:
    def __init__(self):
        self.state = np.zeros((MAX_CARS_COUNT, CAR_ATTRIBUTE_COUNT))
        
         

        self.state[0] =[0,ay,0.1] #agent 

        self.state[1] =[lx1,0,cars_speed]
        self.state[2] =[lx1,WORD_HEIGHT/2,cars_speed]
        self.state[3] =[lx2,WORD_HEIGHT/6,cars_speed]
        self.state[4] =[lx2,5*WORD_HEIGHT/6,cars_speed]

        self.state[5] =[lx3,WORD_HEIGHT/2,-cars_speed]
        self.state[6] =[lx3,WORD_HEIGHT ,-cars_speed]
        
        self.state[7] =[lx4,WORD_HEIGHT/6,-cars_speed]
        self.state[8] =[lx4,3*WORD_HEIGHT/4 ,-cars_speed]
        self.state[9] =[lx4,5*WORD_HEIGHT/6 ,-cars_speed]


        self.score =0 
    def frame_step(self ,input_actions,t):
        #clock.tick(FPS)

        if input_actions[1] == 1:
            # increment speed
            self.state[0,2]+=0.1
            if self.state[0,2] > AGENT_MAX_SPEED :
                self.state[0,2]=AGENT_MAX_SPEED 
        if input_actions[2] == 1:
            # decrement speed
            self.state[0,2]-=0.1
            if self.state[0,2] < 0 :
                self.state[0,2]=0
       


        self.move_cars()

        terminal =False
        reward =1

        if self.state[0,2]==0:
            reward =-1
        
        if self.arrived():
            self.reset_agent()
            reward =100
            self.score +=1 

        if self.detect_collision():
            self.reset_agent()
            terminal =True
            reward =-100
            self.score =0 
	"""
        pygame.display.set_caption(str(self.state[0,2]))

        #drawing
        SCREEN.blit(bg_image, (0,0))
        for car  in self.state:
            SCREEN.blit(car_image, (car[0],car[1]))
        
        
        pygame.display.set_caption(str(self.state[0,2]))
        pygame.display.update()
        #image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        """
        return self.state , reward , terminal , self.score

    def arrived(self):
        car = self.state[0]
        if car[0]>WORD_WIDTH :
            return True

    def reset_agent(self):
        car = self.state[0]
        car[0]=0
        car[2]=0

    def move_cars(self):
        car = self.state[0]
        car[0]+=car[2]
 
            

        for i  in range(1,MAX_CARS_COUNT) :
            car = self.state[i]
            car[1]+=car[2]  
            if not( 0 < car[1] < WORD_HEIGHT )  :
                if car[2] > 0:
                    car[1]= 0
                else:
                    car[1] = WORD_HEIGHT 

 

    
 
 
    

    def detect_collision(self):
        c1 =self.state[0]
        for i  in range(1,MAX_CARS_COUNT) :
            c2 = self.state[i]
            if check_car_collision(c1,c2) :
                return True
        return False


CAR_WIDTH=16
CAR_HEIGHT=16

def check_car_collision(car1, car2):
    return not (car1[0] > car2[0] + CAR_WIDTH or car1[0] + CAR_WIDTH < car2[0] or car1[1] > car2[1]+CAR_HEIGHT or car1[1]+CAR_HEIGHT < car2[1])


 
