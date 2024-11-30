import numpy as np
from tdmclient import ClientAsync, aw

def set_motors(left, right, node):
    aw(node.set_variables(motors(left, right)))

def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

def motorsstop(node):
    aw(node.set_variables(motors(0,0)))

def read_motors_speed(node,client):
    aw(node.wait_for_variables({"motor.left.speed","motor.right.speed"}))
    speed_motors=[node.v.motor.left.speed, node.v.motor.right.speed]
    return speed_motors