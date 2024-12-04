import numpy as np
import local_avoidance
from utils import *
class motion_controller:
    def __init__(self):
        self.control_mode = "path_following"
        self.local_nav = local_avoidance.local_nav()
    def set_mode(self, prox_horizontal,x,y,theta): # x,y,theta from ekf
        recompute_global_path = False
        if self.control_mode == "path_following":
            activate_local_avoidance = local_avoidance.check_obstacle(prox_horizontal)
            if activate_local_avoidance:
                self.control_mode = "local_avoidance"
                self.local_nav.initialize(prox_horizontal)
                print("local avoidance activated")
                self.x_entrance = x
                self.y_entrance = y
                self.alpha_entrance = theta
        elif self.control_mode == "local_avoidance":
            get_back_to_path = self.activate_path_following(x,y, theta)
            if get_back_to_path:
                print("getting back to path activated")
                self.control_mode = "get_back_to_path"
                self.x_exit = x
                self.y_exit = y
                # recompute_global_path = True
        elif self.control_mode == "get_back_to_path":
            #dtheta = (theta - self.alpha_entrance + np.pi) % (2*np.pi) - np.pi
            distance = np.sqrt(x - self.x_exit, y - self.y_exit)
            if distance > 3: # has moved far enough from exit point
                self.control_mode = "path_following"
                recompute_global_path = True
                print("path following activated activated")
        return recompute_global_path
    def activate_path_following(self, x, y, theta):
        dy = y - self.y_entrance
        dx = x - self.x_entrance
        alpha = np.arctan2(dy, dx)
        #d_alpha = (theta - self.alpha_entrance + np.pi) % (2*np.pi) - np.pi
        d_alpha = (alpha - self.alpha_entrance + np.pi) % (2*np.pi) - np.pi
        return ( abs(d_alpha) < 0.1 and np.sqrt(dy**2 + dx**2) > 5 ) # if alpha is small enough, assume that Thymio is back on track
    def find_how_go (self,x,y,theta,x_goal,y_goal):

        k_rho = 0.8  # Le robot avance à une vitesse linéaire modérée
        k_alpha = 0.02  # Ajuste l'orientation rapidement mais sans oscillations
        k_beta = -0.5  # Orientation finale modérée (négatif si la base de contrôle le requiert)
        
        dx = x_goal - x
        dy = y_goal - y
        
        # Conversion en coordonnées polaires
        rho = np.sqrt(dx**2 + dy**2)  # Distance à l'objectif
        alpha = np.arctan2(dy, dx) - theta  # Angle relatif à la cible
        alpha = (alpha + np.pi)%(2*np.pi) - np.pi

        desired_theta = np.arctan2(dy, dx)
        dtheta = (desired_theta - theta + np.pi)% (2*np.pi) - np.pi
        if abs(dtheta) > 3*np.pi / 180: # abs(dtheta) larger than 3 degree
            v_ = 0
            w_ = np.sign(dtheta)*0.314
        else:
            v_ = 3.3
            w_ = 0
        return v_, w_
        
        # # Commandes de contrôle
        # v = k_rho * rho
        # v = min(v, 3.3) # do not go faster than 3.3
        # omega = k_alpha * alpha 
        
        # return v, omega
    
    def compute_control(self, x,y,theta,x_goal, y_goal, prox_horizontal):
        if self.control_mode == "local_avoidance":
            v, w = self.local_nav.calculate_speed_local_nav(prox_horizontal)
        elif self.control_mode == "path_following":
            v,w = self.find_how_go(x,y,theta,x_goal, y_goal)
        elif self.control_mode == "get_back_to_path":
            if self.local_nav.wall_on == "right":
                v, w = 2, 0.314
            else:
                v, w = 2, -0.314
        return from_vw_to_u(v,w)
