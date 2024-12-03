import math
import numpy as np
from utils import *

def check_obstacle(prox_horizontal):
    return any(np.array(prox_horizontal[0:5]) > 800)

class local_nav:
    def __init__(self):
        self.wall_on = "right"
        self.mode = "turning"
    def initialize(self, prox_horizontal):
        if prox_horizontal[0]+ prox_horizontal[1]<=prox_horizontal[3] + prox_horizontal[4]:
            self.wall_on = "right"
            self.mode = "turning"
            print(f"local nav mode = {self.mode}, wall on {self.wall_on}")
        else:
            self.wall_on = "left"
            self.mode = "turning"
            print(f"local nav mode = {self.mode}, wall on {self.wall_on}")

    def calculate_speed_local_nav(self, prox_horizontal):
        if self.mode == "turning":
            if self.wall_on == "right":
                v = -1
                w = 0.314
                if all(np.array(prox_horizontal[2:]) < 150):
                    self.mode = "wall_following"
                    print(f"{self.mode} on {self.wall_on} activated")
            elif self.wall_on == "left":
                v = -1
                w = - 0.314
                if all(np.array(prox_horizontal[0:3]) < 150):
                    self.mode = "wall_following"
                    print(f"{self.mode} on {self.wall_on} activated")
            
        elif self.mode == "wall_following":
            if self.wall_on == "right":
                if all(np.array(prox_horizontal[2:]) < 150): # not seeing the wall on right, turning right
                    v, w = 3, -0.15
                else: # seeing the wall, move and turn slightly left
                    v, w = 3, 0.3
            if self.wall_on == "left":
                if all(np.array(prox_horizontal[0:3]) < 150): # not seeing the wall on left, turning left
                    v, w = 3, 0.15
                else: # seeing the wall, move and turn slightly right
                    v, w = 3, -0.3

        return v,w



# def calculate_speed_local_nav(prox_horizontal):
#     if any(np.array(prox_horizontal[2:])):
#     spLeft = -50
#     spRight= 50

#     if(prox_horizontal[0] >1000 and prox_horizontal[1] > 500):
#         spLeft = 50
#         spRight = -50

#     if (prox_horizontal[0] >1000 and prox_horizontal[1] < 500):
#         spLeft = 100
#         spRight = 100
#     if (prox_horizontal[0] <500 and prox_horizontal[1] <500):
#         spLeft = -50
#         spRight = 50

#     if(prox_horizontal[4] >1000 and prox_horizontal[3] > 500):
#         spLeft = -50
#         spRight = 50

#     if (prox_horizontal[4] >1000 and prox_horizontal[3] < 500):
#         spLeft = 100
#         spRight = 100
#     if (prox_horizontal[4] <500 and prox_horizontal[3] <500):
#         spLeft = 50
#         spRight = -50

#     return spLeft,spRight

