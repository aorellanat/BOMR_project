import cv2
import numpy as np

from path_planning import *


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


def detect_obstacles_and_goal(frame, padding_obstacles, map_width_to_display, map_height_to_display):
    canny_img = preprocess_obstacles(frame)
    mask_obstacles = np.zeros_like(frame)

    obstacle_contours = []
    goal_coords = None

    contours, hierarchy = cv2.findContours(canny_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if hierarchy[0][i][3] != -1 and cv2.contourArea(contour) > 1000:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True).squeeze()

            if len(approx) > 10:
                (x,y), _ = cv2.minEnclosingCircle(contour)
                goal_coords = (int(x), int(y))
            else:
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    centroid = np.array([0, 0])
                else:
                    centroid = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

                mask_obstacle = np.zeros_like(approx)

                for j, vertex in enumerate(approx):
                    vector = vertex - centroid
                    unit_vector = vector / np.linalg.norm(vector)

                    mask_obstacle[j] = vertex + (unit_vector * (padding_obstacles // 2))
                    approx[j] = vertex + unit_vector * padding_obstacles

                cv2.drawContours(mask_obstacles, [mask_obstacle], 0, (255, 255, 255), -1)
                obstacle_contours.append(approx)

    print(f'Number of obstacles: {len(obstacle_contours)}')
    return obstacle_contours, mask_obstacles, goal_coords


def get_obstacle_vertices(obstacles_contours):
    obstacle_vertices = []
    for contour in obstacles_contours:
        for vertex in contour:
            obstacle_vertices.append(vertex)
    return obstacle_vertices


def draw_goal(frame, goal_coords):
    cv2.circle(frame, goal_coords, 7, (0, 255, 0), -1)
    text_coords = (goal_coords[0] + 20, goal_coords[1] + 10)
    cv2.putText(frame, 'Goal', text_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


def draw_obstacles(frame, obstacles_contours):
    img_mask = np.zeros_like(frame)
    map_height, map_width = frame.shape[:2]
    
    for contour in obstacles_contours:
        for vertex in contour:
            cv2.circle(frame, tuple(vertex), 5, (0, 0, 255), -1)


def detect_thymio(frame, draw_aruco=False):
    img_gray = preprocess_map(frame)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected = detector.detectMarkers(img_gray)

    thymio_found = False
    thymio_coords = None
    thymio_angle = None

    if ids is not None:
        for aruco_id, corners in zip(ids, corners):
            aruco_id = np.squeeze(aruco_id)
            if aruco_id == 1:
                if draw_aruco:
                    cv2.aruco.drawDetectedMarkers(frame, [corners], aruco_id)

                corners = np.squeeze(corners)
                tl = corners[0]
                tr = corners[1]
                top_mid = (int((tl[0] + tr[0]) // 2), int((tl[1] + tr[1]) // 2))

                thymio_center_x = int((corners[0][0] + corners[2][0]) // 2)
                thymio_center_y = int((corners[0][1] + corners[2][1]) // 2)

                thymio_coords = (thymio_center_x, thymio_center_y)
                cv2.circle(frame, thymio_coords, 7, (255, 0, 0), -1)

                # Angle calculation, please modify it as you need
                c_o = top_mid[0] - thymio_center_x
                c_a = top_mid[1] - thymio_center_y
                thymio_angle = np.arctan2(c_a, c_o)

                cv2.arrowedLine(frame, thymio_coords, top_mid, (0, 255, 0), 7, tipLength=0.5)
    
                thymio_found = True

    return thymio_found, thymio_coords, thymio_angle


def main():
    # -------- Constants -------- #
    CAMERA_ID = 0

    MAP_MAX_HEIGHT = 600
    MAP_MAX_WIDTH = 800

    MAP_WIDTH_TO_DISPLAY = 500
    MAP_HEIGHT_TO_DISPLAY = 400

    PADDING_OBSTACLES = 30
    # -------- Variables -------- #
    map_detection = False
    obstacles_detection = False
    path_planning = False
    start_motion = False

    map_coords = []
    obstacles_contours = []

    goal_coords = None

    thymio_found = False
    thymio_coords = []
    thymio_angle = None

    path = []

    mask_obstacles = None
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
                obstacles_contours, mask_obstacles, goal_coords = detect_obstacles_and_goal(
                    map_frame,
                    PADDING_OBSTACLES, 
                    MAP_WIDTH_TO_DISPLAY,
                    MAP_HEIGHT_TO_DISPLAY
                )
                obstacles_detection = False
            
            if len(obstacles_contours) > 0:
                draw_obstacles(map_frame, obstacles_contours)

            if goal_coords:
                draw_goal(map_frame, goal_coords)

            thymio_found, thymio_coords, thymio_angle = detect_thymio(map_frame)

            # Step 3: Path planning
            if path_planning:
                if thymio_found and goal_coords:
                    start_array_coord = np.array(thymio_coords)
                    goal_array_coords = np.array(goal_coords)
                    obstacle_vertices = get_obstacle_vertices(obstacles_contours)

                    path = compute_global_path(start_array_coord, goal_array_coords, obstacle_vertices, mask_obstacles)
                else:
                    print(f'It was not possible to detect the path planning. Thymio: {thymio_found}, Goal: {goal_coords}')

                path_planning = False

            # Step 4: Init the project
            # if start_motion:
                # ------> Important: Here you have the position, and angle of the thymio
                # print(f'Thymio coordinates: {thymio_coords}')

            if len(path) > 0:
                draw_path(path, map_frame)

            # Reshape map before display it
            map_frame = cv2.resize(map_frame, (MAP_WIDTH_TO_DISPLAY, MAP_HEIGHT_TO_DISPLAY))
            cv2.imshow('Map', map_frame)

        if thymio_found:
            cv2.putText(frame_copy, f'Thymio (x,y): {thymio_coords}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 4)
            cv2.putText(frame_copy, f'Thymio angle rad: {thymio_angle:.4f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 4)
            cv2.putText(frame_copy, f'Thymio angle deg: {np.degrees(thymio_angle):.4f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 4)

        cv2.imshow('frame', frame_copy)

        # ---------- Keyboard options ---------- #
        # 1. Detect the map, obstacles and goal
        if cv2.waitKey(1) & 0xFF == ord('m'):
            map_detection = True

        # 2. Path planning
        if cv2.waitKey(1) & 0xFF == ord('p'):
            path_planning = True
            print('Key p pressed')

        # 3. Start the project
        if cv2.waitKey(1) & 0xFF == ord('s'):
            start_motion = True

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
