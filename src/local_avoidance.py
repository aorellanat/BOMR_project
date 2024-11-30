import math
import numpy as np


def check_obstacle(prox_horizontal):
    if any(np.array(prox_horizontal[0:5]) > 1000):
        return True
    else:
        return False
class local_nav:
    def __init__(self):
        self.wall_on = "right"
        self.mode = "turning"
    def initialize(self, prox_horizontal):
        if prox_horizontal[1]<=prox_horizontal[3]:
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
                ul = -50
                ur = 50
                if all(np.array(prox_horizontal[2:]) < 500):
                    self.mode = "wall_following"
                    print(f"{self.mode} on {self.wall_on} activated")
            elif self.wall_on == "left":
                ul = 50
                ur = -50
                if all(np.array(prox_horizontal[0:3]) < 500):
                    self.mode = "wall_following"
                    print(f"{self.mode} on {self.wall_on} activated")
            
        elif self.mode == "wall_following":
            if self.wall_on == "right":
                if all(np.array(prox_horizontal[2:]) < 500):
                    ul, ur = 80,50
                else:
                    ul, ur = 80,100
            if self.wall_on == "left":
                if all(np.array(prox_horizontal[0:3]) < 500):
                    ul, ur = 50,80
                else:
                    ul, ur = 100,80
        
        return ul, ur



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

