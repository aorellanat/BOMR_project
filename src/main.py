import time

from cv import *
from path_planning import *
from motion_control import *
from kalman import *
from utils import *
from tdmclient import ClientAsync, aw

# -------- Computer vision constants -------- #
CAMERA_ID = 1

REAL_MAP_HEIGHT_CM = 112
REAL_MAP_WIDTH_CM = 121

MAP_MAX_HEIGHT = 600
MAP_MAX_WIDTH = 800

PADDING_OBSTACLES = 70

# -------- Kalman filter constants -------- #
dt = 0.1
epsilon = 0.5 # to check if thymio is close enough to next target
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
        aw(self.node.wait_for_variables({"motor.left.speed","motor.right.speed", "prox.ground.delta", "prox.horizontal", "acc"}))
        return [self.node["motor.left.speed"],
                self.node["motor.right.speed"],
                list(self.node["prox.ground.delta"]),
                list(self.node["prox.horizontal"]),
                list(self.node["acc"])]

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
    kidnapping = False

    global_path = [] # this used for drawing the path
    path = [] # this is modified with the motion controller and kalman

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
                if kidnapping:
                    aw(thymio.client.sleep(dt))
                    data = thymio.get_data()
                    prox_ground_delta = data[2]
                    print(f"kidnapping, {prox_ground_delta[0]}")
                    if prox_ground_delta[0] > 700:
                        kidnapping = False
                        print("kidnapping finished")
                        continue
                if thymio_found and goal_coords and not kidnapping:
                    obstacle_vertices = get_obstacle_vertices(obstacles_contours)

                    global_path = compute_global_path(thymio_coords, goal_coords, obstacle_vertices, mask_obstacles)
                    if global_path is None:
                        cv2.putText(map_frame, f'Thymio inside obstacle, place again', (500,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        continue
                    path = global_path.copy().tolist()
                    path = [convert_pixel_to_cm(target, REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT) for target in path]
                    next_target = path.pop(0)
                    # Initialize the Kalman filter, need to run only once
                    if not kalman_filter_initialized:
                        thymio_coords_cm = convert_pixel_to_cm(thymio_coords, REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                        initial_state = np.array([thymio_coords_cm[0], thymio_coords_cm[1], thymio_angle, 0, 0]) # x, y, theta, v, w
                        ekf.initialize_X(initial_state)
                        kalman_filter_initialized = True
                    start_motion=True
                    path_planning = False
                    mc.control_mode = "path_following"
                else:
                    # print(f'Trying to find path, Thymio found {thymio_found}, Kidnapping {kidnapping}')
                    cv2.putText(map_frame, f'Trying to find path, Thymio found {thymio_found}, Kidnapping {kidnapping}', (400,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    start_motion=False
                


            # Step 3: Start motion
            if start_motion:
                if next_target:
                    thymio_coords_cm = convert_pixel_to_cm(thymio_coords, REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                    x_camera, y_camera, theta_camera = thymio_coords_cm[0], thymio_coords_cm[1], thymio_angle
                    camera_blocked = not thymio_found

                    ekf.switch_mode(camera_blocked)

                    data = thymio.get_data()
                    ul = data[0]
                    ur = data[1]
                    prox_ground_delta = data[2]
                    prox_horizontal = data[3]
                    acc = data[4]

                    v,w = from_u_to_vw(ul, ur)
                    u = np.array([v,w])
                    z = np.array([x_camera, y_camera, theta_camera, v, w])

                    #use acc to detect kidnapping, ground sensor to see if Thymio has been put back on floor
                    if abs(acc[2]-22)>4: # sudden change in acc_z indicates kidnapping
                        print("kidnapping detected")
                        kidnapping = True
                        thymio.node.send_set_variables(motors(0,0))
                        # computes new global path in the next iteration
                        path_planning = True
                        continue

                    # updating thymio state
                    ekf.predict_and_update(u,z)
                    x,y,theta,v,w = ekf.get_X()
                    ekf_pixel = convert_cm_to_pixel((x,y), REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                    arrow_end = convert_cm_to_pixel((x + 5*np.cos(theta), y + 5*np.sin(theta)), REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                    x_target, y_target = next_target

                    #draw estimated state and variance of thymio
                    cv2.circle(map_frame, ekf_pixel, 7, (255, 0, 255), -1)
                    cv2.arrowedLine(map_frame, ekf_pixel, arrow_end, (255,0,255), 7, tipLength=0.5)
                    xy_variance = ekf.P[0:2, 0:2]
                    eigenvalues, eigenvectors = np.linalg.eig(xy_variance)
                    ellipse_axis_length = convert_cm_length_to_pixel(eigenvalues,REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                    ellipse_angle = np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
                    ellipse_angle = np.rad2deg(-ellipse_angle)
                    cv2.ellipse(map_frame, ekf_pixel, ellipse_axis_length, ellipse_angle, 0, 360, (0,255,255), 5)

                    #  check if next_target is reached
                    target_reached = np.linalg.norm(np.array([x - next_target[0], y - next_target[1]])) < epsilon
                    if target_reached:
                        if len(path) == 0:
                            print("Goal reached, terminating...")
                            break
                        else:
                            print("Heading towards next waypoint")
                            next_target = path.pop(0)

                    # in the end, should return desired v and w
                    mc.set_mode(prox_horizontal, x, y, theta)
                    ul, ur = mc.compute_control(x, y, theta, next_target[0], next_target[1], prox_horizontal)
                    if (mc.control_mode == "get_back_to_path"):
                        path_planning = True

                    cv2.putText(map_frame, f'Camera: {x_camera:.2f},{y_camera:.2f}, {theta_camera:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(map_frame, f'EKF   : {x:.2f},{y:.2f}, {theta:.2f}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(map_frame, f'Control input: {ul}, {ur}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(map_frame, f'Thymio found from camera: {thymio_found}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    if mc.control_mode == "path_following":
                        cv2.putText(map_frame, f'{mc.control_mode}, Next target: {x_target:.2f}, {y_target:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    elif mc.control_mode == "local_avoidance":
                        if mc.local_nav.mode =="turning":
                            cv2.putText(map_frame, f'{mc.control_mode}, Obstacle detcted on {mc.local_nav.wall_on}, turning', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        if mc.local_nav.mode =="wall_following":
                            cv2.putText(map_frame, f'{mc.control_mode}, Trying to move along the wall on {mc.local_nav.wall_on}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(map_frame, f'Entrance angle: {mc.alpha_entrance:.2f}, current angle {theta:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    thymio.node.send_set_variables(motors(ul, ur))
                    # if counter % 10 == 0:s
                    #     print(f"motor input = {next_target[0]:.2f}, {next_target[1]:.2f}")
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
        aw(thymio.node.unlock())
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
