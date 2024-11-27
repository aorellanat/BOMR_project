import math

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

def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

def set_motors(left, right, node):
    aw(node.set_variables(motors(left, right)))

def check_obstacle(prox_horizontal):
    if prox_horizontal[0] > 1000 or prox_horizontal[2] > 1000 or prox_horizontal[1] > 1000:
        state = 2
        return True
    else:
        return False

def calculate_speed_local_nav(prox_horizontal,speed0):
    
    spLeft = speed0
    spRight = speed0

    if prox_horizontal[0] > 1000 and (prox_horizontal[1] > 500 or prox_horizontal[2] > 500):
        spLeft = 100
        spRight = -100

    elif prox_horizontal[0] > 1000 and prox_horizontal[1] < 500 and prox_horizontal[2] < 500:
        spLeft = 100
        spRight = 100

    elif prox_horizontal[0] < 500 and prox_horizontal[1] < 500:
        spLeft = -100
        spRight = 100

    return[spLeft,spRight]


def find_position (p_new,p_old,spLeft,spRight):

    # Save the previous position
    p_old[0], p_old[1], p_old[2] = p_new

    sampling_time = 10  # in seconds

    d_left = spLeft * sampling_time / 100
    d_right = spRight * sampling_time / 100

    d = (d_left + d_right) / 2
    delta_theta = (d_right - d_left) / wheel_distance

    p_new[0] += d * math.cos(p_new[2] + delta_theta / 2)
    p_new[1] += d * math.sin(p_new[2] + delta_theta / 2)
    p_new[2] = (p_new[2] + math.pi) % (2 * math.pi) - math.pi

    # Adjust motor speeds
    diffDelta = (p_new[0] + p_new[1]) - (p_old[0] + p_old[1])
    spLeft = speed0 - speedGain * diffDelta
    spRight = speed0 + speedGain * diffDelta

    return {spLeft,spRight}

