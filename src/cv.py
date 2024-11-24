import cv2
import numpy as np

from node import *


def preprocess_map(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_median = cv2.medianBlur(img_gray, 3)
    bilateral_img = cv2.bilateralFilter(img_median, 9, 75, 75)
    return bilateral_img


def preprocess_obstacles(frame):
    img_gray = preprocess_map(frame)
    canny_img = cv2.Canny(img_gray, 50, 150)
    return canny_img


def detect_map(frame, map_max_width, map_max_height, draw_arucos=False):
    print('Detecting map...')
    img_gray = preprocess_map(frame)

    # Create the aruco dictionary and detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(img_gray)

    map_coords = np.zeros((4, 2), dtype=np.float32)

    if ids is not None:
        if draw_arucos:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for aruco_id, corners in zip(ids, corners):
            aruco_id = np.squeeze(aruco_id)
            corners = np.squeeze(corners)
            
            # Corners are ordered in clockwise order, and start from the top left corner
            if aruco_id == 2:
                map_coords[0] = corners[0]
            elif aruco_id == 3:
                map_coords[1] = corners[1]
            elif aruco_id == 4:
                map_coords[2] = corners[2]
            elif aruco_id == 5:
                map_coords[3] = corners[3]
            else:
                continue

        return map_coords


def detect_obstacles_and_goal(frame):
    img_gray = preprocess_obstacles(frame)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    obstacle_contours = []
    goal_coords = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        print(f'Approx: {len(approx)}')

        # If the shape is a circle, it is the goal
        if len(approx) > 8:
            M = cv2.moments(contour)
            goal_coords = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        else:
            obstacle_contours.append(contour)

    return obstacle_contours, goal_coords


def draw_goal(frame, goal_coords, grid_size):
    goal_coords = (goal_coords[0] // grid_size, goal_coords[1] // grid_size)
    cv2.circle(frame, goal_coords, 20, (0, 255, 0), -1)
    cv2.putText(frame, 'Goal', goal_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)


def draw_obstacles(frame, obstacles_contours, grid_size, draw_contours=False):
    img_mask = np.zeros_like(frame)

    map_height, map_width = frame.shape[:2]
    n_rows = map_height // grid_size
    n_columns = map_width // grid_size

    grid = np.zeros((n_rows, n_columns), dtype=np.uint8)
    
    for contour in obstacles_contours:
        cv2.drawContours(img_mask, [contour], -1, 255, -1)

    if draw_contours:
        cv2.drawContours(frame, obstacles_contours, -1, (0, 255, 0), 2)
        cv2.imshow('mask', img_mask)

    for i in range(n_rows):
        for j in range(n_columns):
            x_start, y_start = j * grid_size, i * grid_size
            x_end, y_end = x_start + grid_size, y_start + grid_size

            cell_mask = img_mask[y_start:y_end, x_start:x_end]
            if np.any(cell_mask):
                grid[i, j] = 1
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), -1)
            else:
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 0), 1)

    return grid


def detect_thymio(frame):
    img_gray = preprocess_map(frame)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(img_gray)

    thymio_coords = None
    thymio_angle = None

    if ids is not None:
        for aruco_id, corners in zip(ids, corners):
            aruco_id = np.squeeze(aruco_id)
            if aruco_id == 1:
                cv2.aruco.drawDetectedMarkers(frame, [corners], aruco_id)

                corners = np.squeeze(corners)

                thymio_center_x = int((corners[0][0] + corners[2][0]) // 2)
                thymio_center_y = int((corners[0][1] + corners[2][1]) // 2)

                thymio_coords = (thymio_center_x, thymio_center_y)
                cv2.circle(frame, thymio_coords, 20, (0, 0, 255), -1)

                # Angle calculation, please modify it as you need
                c_o = thymio_center_x - corners[0][0]
                c_a = thymio_center_y - corners[0][1]
                thymio_angle = np.arctan2(c_a, c_o)

    if thymio_coords:
        cv2.putText(frame, f'Thymio (x,y): {thymio_coords}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, f'Thymio angle rad: {thymio_angle}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
        cv2.putText(frame, f'Thymio angle deg: {np.degrees(thymio_angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)

    return thymio_coords, thymio_angle


def main():
    # -------- Constants -------- #
    CAMERA_ID = 0

    MAP_MAX_HEIGHT = 600
    MAP_MAX_WIDTH = 800
    GRID_SIZE = 30 # Size of a grid cell in pixels

    MAP_WIDTH_TO_DISPLAY = 500
    MAP_HEIGHT_TO_DISPLAY = 400

    # -------- Variables -------- #
    map_detection = False
    obstacles_detection = False
    path_planning = False
    start_motion = False

    map_coords = []
    obstacles_contours = []
    grid = np.zeros((MAP_MAX_HEIGHT // GRID_SIZE, MAP_MAX_WIDTH // GRID_SIZE))
    print(f'Grid shape: \nRows: {grid.shape[0]}\nColumns: {grid.shape[1]}')

    goal_coords = None
    thymio_coords = None
    thymio_angle = None

    frame_copy = None

    camera = cv2.VideoCapture(CAMERA_ID)
    # -------- Main loop -------- #
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_copy = frame.copy()

        # Step 1: Detect the map
        if map_detection:
            map_coords = detect_map(frame, MAP_MAX_WIDTH, MAP_MAX_HEIGHT, draw_arucos=True)
            if len(map_coords) == 0:
                print(f'No map detected: only {len(map_coords)} corners found')

            map_detection = False
            obstacles_detection = True

        if len(map_coords) == 4:
            # Draw the map contour
            cv2.polylines(frame_copy, [map_coords.astype(np.int32)], True, (0, 255, 0), 2)

            # Perspective transformation
            pts2 = np.float32([[0, 0], [MAP_MAX_WIDTH, 0], [MAP_MAX_WIDTH, MAP_MAX_HEIGHT], [0, MAP_MAX_HEIGHT]])
            matrix = cv2.getPerspectiveTransform(map_coords, pts2)
            map_frame = cv2.warpPerspective(frame, matrix, (MAP_MAX_WIDTH, MAP_MAX_HEIGHT))

            # Step 2: Detect the obstacles inside the map and the goal
            if obstacles_detection:
                obstacles_contours, goal_coords = detect_obstacles_and_goal(map_frame)
                obstacles_detection = False
            
            if len(obstacles_contours) > 0:
                grid = draw_obstacles(map_frame, obstacles_contours, GRID_SIZE)

            if goal_coords:
                draw_goal(map_frame, goal_coords, GRID_SIZE)

            # Step 3: Path planning
            if path_planning:
                start_coords, _ = detect_thymio(map_frame)

                start_node_cell = (start_coords[1] // GRID_SIZE, start_coords[0] // GRID_SIZE)
                goal_node_cell = (goal_coords[1] // GRID_SIZE, goal_coords[0] // GRID_SIZE)

                if start_coords and goal_coords:
                    start_node = Node(start_node_cell)
                    goal_node = Node(goal_node_cell)

                    # Call path planning here: <------ Path planning
                    # astar_path = astar(start_node, goal_node, mask_obstacles, nodes)
                    # draw_path(astar_path)

                path_planning = False

            # Step 4: Detect the Thymio
            if start_motion:
                thymio_coords, thymio_angle = detect_thymio(map_frame) # ------> Important: Here you have the position, and angle of the thymio

            # Reshape map before display it
            map_frame = cv2.resize(map_frame, (MAP_WIDTH_TO_DISPLAY, MAP_HEIGHT_TO_DISPLAY))
            cv2.imshow('Map', map_frame)

        cv2.imshow('frame', frame_copy)

        # ---------- Keyboard options ---------- #
        # 1. Detect the map, obstacles and goal
        if cv2.waitKey(1) & 0xFF == ord('m'):
            print('Key m pressed')
            map_detection = True

        # 2. Path planning
        if cv2.waitKey(1) & 0xFF == ord('p'):
            path_planning = True
            print('Key p pressed')

        # 3. Start the project
        if cv2.waitKey(1) & 0xFF == ord('s'):
            start_motion = True
            print('Key s pressed')

        # 4. Quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Closing the program...')
            break

        # 5. Reset variables
        if cv2.waitKey(1) & 0xFF == ord('r'):
            print('Resetting variables...')
            map_coords = []
            obstacles_contours = []
            goal_coords = None
            thymio_coords = None
            thymio_angle = None
            start_motion = False
            path_planning = False
            map_detection = False

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
