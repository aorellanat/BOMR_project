import time

from cv import *
from path_planning import *
from motion_control import *
from kalman import *
from utils import *
from tdmclient import ClientAsync, aw

# -------- Computer vision constants -------- #
CAMERA_ID = 0

REAL_MAP_HEIGHT_CM = 84
REAL_MAP_WIDTH_CM = 88.5

MAP_MAX_HEIGHT = 600
MAP_MAX_WIDTH = 800

PADDING_OBSTACLES = 30

# -------- Kalman filter constants -------- #
dt = 0.1
epsilon = 0.7 # to check if thymio is close enough to next target
state_dim=5
measurement_dim=5
control_dim=2


# ------ Utility function to set the motors information ------ #å
def motors(left, right):
    return {
        "motor.left.target": [left],
        "motor.right.target": [right],
    }

# -------- Functions to communicate with the Thymio -------- #
class Thymio:
    def __init__(self):
        self.client = ClientAsync()
        self.node = aw(self.client.wait_for_node())
        aw(self.node.lock())
        print('Connected to Thymio!')

    def get_data(self):
        aw(self.node.wait_for_variables({"motor.left.speed","motor.right.speed", "acc", "prox.horizontal"}))
        return [self.node["motor.left.speed"],
                self.node["motor.right.speed"],
                list(self.node["acc"]),
                list(self.node["prox.horizontal"])]

    def set_motors(self, left, right):
        self.node.send_set_variables(motors(left, right))


def main():
    # -------- Variables -------- #
    map_detection = False
    obstacles_detection = False
    path_planning = False
    start_motion = False
    kalman_filter_initialized = False

    map_coords = []
    obstacles_contours = []
    mask_obstacles = None

    thymio_found = False
    thymio_coords = [0, 0]
    thymio_angle = 0
    goal_coords = None

    global_path = []
    path = []

    ekf = ExtendedKalmanFilter(state_dim, measurement_dim, control_dim,dt)
    mc = motion_controller()
    counter = 0
    camera_blocked = False
    next_target = [0,0]

    try:
        camera = cv2.VideoCapture(CAMERA_ID)
        thymio = Thymio()

        # -------- Main loop -------- #
        while True:
            ret, frame = camera.read()
            if not ret:
                break

            # Step 1: Detect the map
            if map_detection:
                map_coords = detect_map(frame, MAP_MAX_WIDTH, MAP_MAX_HEIGHT, draw_arucos=True)
                if len(map_coords) == 0:
                    print(f'No map detected: only {len(map_coords)} corners found')

                map_detection = False
                obstacles_detection = True

            if len(map_coords) == 4:
                # Draw the map contour
                cv2.polylines(frame, [map_coords.astype(np.int32)], True, (0, 255, 0), 2)

                # Perspective transformation
                map_frame = fix_map_perspective(frame, map_coords, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)

                # Detect the obstacles inside the map and the goal
                if obstacles_detection:
                    obstacles_contours, mask_obstacles, goal_coords = detect_obstacles_and_goal(map_frame, PADDING_OBSTACLES)
                    obstacles_detection = False

                # Every iteration check if thymio is detected and get its information
                thymio_found, thymio_coords, thymio_angle = detect_thymio(map_frame)


            # Step 2: Path planning
            if path_planning:
                if thymio_found and goal_coords:
                    obstacle_vertices = get_obstacle_vertices(obstacles_contours)
                    global_path = compute_global_path(thymio_coords, goal_coords, obstacle_vertices, mask_obstacles)
                    path = global_path.copy().tolist()

                    # Initialize the Kalman filter, need to run only once
                    if not kalman_filter_initialized:
                        thymio_coords_cm = convert_pixel_to_cm(thymio_coords, REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                        initial_state = np.array([thymio_coords_cm[0], thymio_coords_cm[1], thymio_angle, 0, 0]) # x, y, theta, v, w
                        ekf.initialize_X(initial_state)
                        kalman_filter_initialized = True
                        next_target = path.pop(0)
                else:
                    print(f'No path found. Thymio: {thymio_found}, Goal: {goal_coords}')
                path_planning = False


            # Step 3: Start motion
            if start_motion:
                if len(path) > 0:
                    thymio_coords_cm = convert_pixel_to_cm(thymio_coords, REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                    x_camera, y_camera, theta_camera = thymio_coords_cm[0], thymio_coords_cm[1], thymio_angle
                    camera_blocked = not thymio_found

                    ekf.switch_mode(camera_blocked)

                    data = thymio.get_data()
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
                        thymio.node.send_set_variables(motors(0,0))
                        aw(client.sleep(3))
                        # computes new global path in the next iteration
                        path_planning = True
                        continue

                    # updating thymio state
                    ekf.predict_and_update(u,z)
                    x,y,theta,v,w = ekf.get_X()
                    if counter % 10 == 0:
                        print(f"state = {x:.2f}, {y:.2f}, {theta:.2f}, {v:.2f}, {w:.2f}")
                    
                    #  check if next_target is reached
                    target_reached = np.linalg.norm(np.array([x - next_target[0], y - next_target[1]])) < epsilon
                    if target_reached:
                        if len(path) == 0:
                            print("Goal reached, terminating...")
                            break
                        else:
                            print("Heading towards next waypoint")
                            next_target = path.pop(0)
                            print(f"next_target = {next_target[0]}, {next_target[1]}")

                    # in the end, should return desired v and w
                    mc.set_mode(prox_horizontal, x, y, theta)
                    ul, ur = mc.compute_control(x, y, theta,next_target[0], next_target[1], prox_horizontal)
                    thymio.node.send_set_variables(motors(ul, ur))
                    aw(thymio.client.sleep(dt))
                    counter += 1
                else:
                    print('No path to follow')


            # ---------- Display the frames ---------- #
            if len(obstacles_contours) > 0:
                draw_obstacles(map_frame, obstacles_contours)

            if goal_coords:
                draw_goal(map_frame, goal_coords)

            if len(global_path) > 0:
                draw_path(global_path, map_frame)

            if len(map_coords) == 4:
                # map_frame = cv2.resize(map_frame, (500, 400))
                cv2.imshow('Map', map_frame)

            cv2.imshow('Frame', frame)

            # ---------- Keyboard options ---------- #
            if cv2.waitKey(1) & 0xFF == ord('m'):
                print('Detecting map...')
                map_detection = True

            if cv2.waitKey(1) & 0xFF == ord('p'):
                print('Path planning...')
                path_planning = True

            if cv2.waitKey(1) & 0xFF == ord('s'):
                print('Start motion...')
                start_motion = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Closing the program...')
                break


        thymio.set_motors(0, 0)
        camera.release()
        cv2.destroyAllWindows()
        thymio.node.send_set_variables(motors(0, 0))
        aw(thymio.node.unlock())
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
