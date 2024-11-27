import numpy as np
import control as contr

class AstolfiController:
    def __init__(self, k_rho, k_alpha, k_beta):
        """
        Initialisation du contrôleur Astolfi.
        
        :param k_rho: Gain pour la distance à l'objectif
        :param k_alpha: Gain pour l'orientation relative
        :param k_beta: Gain pour l'orientation finale
        """
        self.k_rho = k_rho
        self.k_alpha = k_alpha
        self.k_beta = k_beta

def find_how_go (self,x,y,theta,x_goal,y_goal)
    
    dx = x_goal - x
    dy = y_goal - y
    
    # Conversion en coordonnées polaires
    rho = np.sqrt(dx**2 + dy**2)  # Distance à l'objectif
    alpha = np.arctan2(dy, dx) - theta  # Angle relatif à la cible
    beta = -theta - alpha  # Orientation finale relative
    
    # Ajustement des angles pour rester dans [-pi, pi]
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    beta = np.arctan2(np.sin(beta), np.cos(beta))
    
    # Commandes de contrôle
    v = self.k_rho * rho
    omega = self.k_alpha * alpha + self.k_beta * beta
    
    return v, omega

def go_to_point (self,x,y,theta,x_goal,y_goal)
    
    [v,omega] = find_how_go (self,x,y,theta,x_goal,y_goal)
    
    v_left = v - (omega * self.wheel_base / 2)
    v_right = v + (omega * self.wheel_base / 2)

    contr.set_motors(v_left,v_right)

    return