import numpy as np
import control as contr


def find_how_go (x,y,theta,x_goal,y_goal):

    k_rho = 0.8  # Le robot avance à une vitesse linéaire modérée
    k_alpha = 1.0  # Ajuste l'orientation rapidement mais sans oscillations
    k_beta = -0.5  # Orientation finale modérée (négatif si la base de contrôle le requiert)
    
    dx = x_goal - x
    dy = y_goal - y
    
    # Conversion en coordonnées polaires
    rho = np.sqrt(dx**2 + dy**2)  # Distance à l'objectif
    alpha = np.arctan2(dy, dx) - theta  # Angle relatif à la cible
    #beta = -theta - alpha  # Orientation finale relative
    
    # Ajustement des angles pour rester dans [-pi, pi]
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
    #beta = np.arctan2(np.sin(beta), np.cos(beta))
    
    # Commandes de contrôle
    v = k_rho * rho

    v = min(100,v)
    omega = k_alpha * alpha 


    v = int(v)
    
    return [v, omega]

def go_to_point (x,y,theta,x_goal,y_goal,node):
    
    [v,omega] = find_how_go(x,y,theta,x_goal,y_goal)

    if (abs(omega) > 0.3):
        v_left =  -20*int((omega * 9 / 2))
        v_right = 20*int((omega * 9 / 2))
    else:
        v_left =  10 + int(v)
        v_right = 10 + int(v)
    
    contr.set_motors(v_left,v_right,node)

    return