import cv2
import numpy as np
import math


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


def fix_map_perspective(frame, map_coords, map_max_width, map_max_height):
    pts2 = np.float32([[0, 0], [map_max_width, 0], [map_max_width, map_max_height], [0, map_max_height]])
    matrix = cv2.getPerspectiveTransform(map_coords, pts2)
    map_frame = cv2.warpPerspective(frame, matrix, (map_max_width, map_max_height))
    return map_frame


def detect_obstacles_and_goal(frame, padding_obstacles):
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
    thymio_coords = [0, 0]
    thymio_angle = 0

    if ids is not None:
        for aruco_id, corners in zip(ids, corners):
            aruco_id = np.squeeze(aruco_id)
            if aruco_id == 1:
                if draw_aruco:
                    cv2.aruco.drawDetectedMarkers(frame, [corners], aruco_id)

                corners = np.squeeze(corners)
                # An aruco has 4 corners ordered in clockwise order, and start from the top left corner
                # Each corner is an array of 2 elements: x and y coordinates
                tl = corners[0] # top left
                tr = corners[1] # top right
                br = corners[2]	# bottom right
                bl = corners[3]	# bottom left

                # Compute the thymio center point by averaging the top left and bottom right corners
                thymio_center_x = int((corners[0][0] + corners[2][0]) / 2)
                thymio_center_y = int((corners[0][1] + corners[2][1]) / 2)
                thymio_coords = [thymio_center_x, thymio_center_y]
                cv2.circle(frame, tuple(thymio_coords), 7, (255, 0, 0), -1)

                #  ---------------- Please modify the angle calculation as needed ----------------
                # Angle cuadrant 1 should be 1: 0-90, cuadrant 2: 90-180, cuadrant 3: 180-270, cuadrant 4: 270-360
                # Reference: https://github.com/anish-natekar/OpenCV_ArUco_Angle_Estimation/blob/main/aruco_library.py

                # Compute the top middle point
                top_middle = (int((tl[0] + tr[0]) / 2), -int((tl[1] + tr[1]) / 2))

                # Compute the center of the aruco with negative y axis
                centre = (tl[0] + tr[0] + bl[0] + br[0]) / 4, -((tl[1] + tr[1] + bl[1] + br[1]) / 4)
                try:
                    thymio_angle = math.degrees(np.arctan((top_middle[1] - centre[1]) / (top_middle[0] - centre[0])))
                except:
                    # Tangent has asymptote at 90 and 270 degrees
                    if top_middle[1] > centre[1]:
                        thymio_angle = 90
                    elif top_middle[1] < centre[1]:
                        thymio_angle = 270
                
                if top_middle[0] >= centre[0] and top_middle[1] < centre[1]:
                    thymio_angle = 360 + thymio_angle
                elif top_middle[0] < centre[0]:
                    thymio_angle = 180 + thymio_angle

                top_middle = (top_middle[0], -top_middle[1])
                cv2.arrowedLine(frame, tuple(thymio_coords), top_middle, (0, 255, 0), 7, tipLength=0.5)
                # cv2.putText(frame, f'Thymio (x,y): {thymio_coords}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                # cv2.putText(frame, f'Thymio angle in degrees: {thymio_angle:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                thymio_found = True
                thymio_angle = math.radians(thymio_angle)

    return thymio_found, thymio_coords, thymio_angle


def convert_pixel_to_cm(pixel_coords, real_map_width, real_map_height, map_width_pixels, map_height_pixels):
    x, y = pixel_coords
    x_cm = (x * real_map_width) / map_width_pixels
    y_cm = ((map_height_pixels - y) * real_map_height) / map_height_pixels
    return x_cm, y_cm

def convert_cm_to_pixel(cm_coords, real_map_width, real_map_height, map_width_pixels, map_height_pixels):
    x_cm, y_cm = cm_coords
    x = (x_cm * map_width_pixels) / real_map_width
    y = map_height_pixels - (y_cm * map_height_pixels) / real_map_height
    return int(x), int(y)
