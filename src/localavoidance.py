import math

obst = [0, 0, 0, 0, 0]
local_motors_speed = [0, 0]
speed0 = 100
p_old = [0, 0, 0]
p_new = [0, 0, 0]
wheel_distance = 25
import numpy as np

obst = [0, 0, 0, 0, 0]
local_motors_speed = [0, 0]
speed0 = 100
p_old = [0, 0, 0]
p_new = [0, 0, 0]
wheel_distance = 25
speedGain = 1.5
state = 2

async def read_prox_sensors(node, client):
    await node.wait_for_variables({"prox.horizontal"})
    return list(node.v.prox.horizontal)


def check_obstacle(prox_horizontal):
    if prox_horizontal[0] > 1000 or prox_horizontal[2] > 1000 or prox_horizontal[1] > 1000 or prox_horizontal[3] > 1000 or prox_horizontal[4] > 1000:
        return True
    else:
        return False

def calculate_speed_local_nav(prox_horizontal):

    spLeft = 0
    spRight= 0

    if(prox_horizontal[0] >1000 and prox_horizontal[1] > 500):
        spLeft = 100
        spRight = -100

    if (prox_horizontal[0] >1000 and prox_horizontal[1] < 500):
        spLeft = 200
        spRight = 200
    if (prox_horizontal[0] <500 and prox_horizontal[1] <500):
        spLeft = -100
        spRight = 100 

    if(prox_horizontal[4] >1000 and prox_horizontal[3] > 500):
        spLeft = -100
        spRight = 100

    if (prox_horizontal[4] >1000 and prox_horizontal[3] < 500):
        spLeft = 200
        spRight = 200
    if (prox_horizontal[4] <500 and prox_horizontal[3] <500):
        spLeft = 100
        spRight = -100 

    return[spLeft,spRight]


def find_position(x_old, y_old, theta_old, vLeft, vRight, wheel_distance, sampling_time=0.1):
    """
    Met à jour la position et l'orientation d'un robot différentiel.
    
    Paramètres :
    - x_old, y_old, theta_old : position et orientation précédente.
    - vLeft, vRight : vitesses des roues gauche et droite (en cm/s).
    - wheel_distance : distance entre les deux roues (en cm).
    - sampling_time : intervalle de temps entre deux mises à jour (en secondes).
    
    Retourne :
    - x_new, y_new, theta_new : nouvelle position et orientation.
    """
    # Calcul des distances parcourues par les roues
    d_left = vLeft/100*3.3 * sampling_time
    d_right = vRight/100*3.3 * sampling_time
    print("voici vright",vRight)
    print("voici vleft",vLeft)


    # Calcul du déplacement linéaire moyen et de la variation d'angle
    d = (d_left + d_right) / 2
    delta_theta = (d_right - d_left) / wheel_distance/2

    # Mise à jour des coordonnées
    x_new = int(x_old + d * np.cos(theta_old + delta_theta / 2))
    y_new = int(y_old + d * np.sin(theta_old + delta_theta / 2))
    theta_new = (theta_old + delta_theta * sampling_time)

    print("positionx",x_new)
    print("positiony",y_new)
    print("angle",theta_new)



    return x_new, y_new, theta_new


