from tdmclient import ClientAsync, aw
from computer_vision import ComputerVision
from kalman import *
from motion_control import *
from utils import *
from path_planning import *
import time
from threading import Timer
from collections import deque


def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

def get_data():
    aw(node.wait_for_variables({"motor.left.speed","motor.right.speed", "acc", "prox.horizontal"}))
    return [node["motor.left.speed"],
            node["motor.right.speed"],
            list(node["acc"]),
            list(node["prox.horizontal"])]

#connect to thymio
client = ClientAsync()
node = aw(client.wait_for_node())
aw(node.lock())

#main loop
#set parameters here
dt = 0.1
epsilon = 0.7 # to check if thymio is close enough to next target
state_dim=5
measurement_dim=5
control_dim=2
# TODO:connect to camera
computer_vision = ComputerVision()
global_path = computer_vision.map_obstacle_detection()

# initialize ekf with information from cv
ekf = ExtendedKalmanFilter(state_dim, measurement_dim, control_dim,dt)
camera_blocked = True
initial_state = np.array([0,0,np.pi/2,0,0]) # x,y,theta,v,w
ekf.initialize_X(initial_state)

mc = motion_controller()
# main loop
next_target = global_path.popleft()
print(f"next_target = {next_target[0]}, {next_target[1]}")
target_reached = False
counter = 0
start_time = time.time()
while (True):
    #TODO: get data from camera(thymio_found, x, y, theta)
    thymio_found, x_camera, y_camera, theta_camera = computer_vision.get_thymio_info()
    camera_blocked = not thymio_found

    ekf.switch_mode(camera_blocked)

    data = get_data()
    ul = data[0]
    ur = data[1]
    acc_z = data[2][2]
    prox_horizontal = data[3]

    v,w = from_u_to_vw(ul, ur)
    u = np.array([v,w])
    z = np.array([x_camera, y_camera, theta_camera, v, w])

    #if kidnapping detected(from acceleration), stop for 3 sec, compute global path again, continue;
    if abs(acc_z - 22) > 3: # sudden change in acc_z indicates kidnapping
        print("kidnapping detected")
        node.send_set_variables(motors(0,0))
        aw(client.sleep(3))
        #TODO: compute new global path
        global_path = computer_vision.map_obstacle_detection()
        next_target = global_path.popleft()
        continue

    # updating thymio state, obviously
    ekf.predict_and_update(u,z)
    x,y,theta,v,w = ekf.get_X()
    if counter % 15 == 0:
        print(f"state = {x:.2f}, {y:.2f}, {theta:.2f}, {v:.2f}, {w:.2f}")
    
    #  check if next_target is reached
    target_reached = np.linalg.norm(np.array([x - next_target[0], y - next_target[1]])) < epsilon
    if target_reached:
        if len(global_path)==0:
            print("goal reached, terminating")
            break
        else:
            print("heading towards next waypoint")
            next_target = global_path.popleft()
            print(f"next_target = {next_target[0]}, {next_target[1]}")

    #in the end, should return desired v and w
    mc.set_mode(prox_horizontal, x, y, theta)
    ul, ur = mc.compute_control(x,y,theta,next_target[0], next_target[1], prox_horizontal)
    node.send_set_variables(motors(ul, ur))
    aw(client.sleep(dt))
    counter += 1

    # whatever happens, stop after 120 sec
    elapsed_time = time.time() - start_time
    if elapsed_time > 120:
        break

# stop thymio
node.send_set_variables(motors(0,0))
aw(node.unlock())
print("terminating main")