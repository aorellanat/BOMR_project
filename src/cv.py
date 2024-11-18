import cv2
import numpy as np

from node import *

# ----------------- #
# TODO:
# 4. Detect Thymio robot (position and orientation)
# 5. Detect goal coordinates
# 6. Merge with path planning (A* algorithm)
# 7. Document the code
# 8. Add code to replicate the results with and image
# ----------------- #
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


def detect_obstacles(frame):
    img_gray = preprocess_obstacles(frame)
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_obstacles(frame, obstacles_contours, grid_size, draw_contours=False):
    img_mask = np.zeros_like(frame)
    
    for contour in obstacles_contours:
        cv2.drawContours(img_mask, [contour], -1, 255, -1)

    if draw_contours:
        cv2.drawContours(frame, obstacles_contours, -1, (0, 255, 0), 2)
        cv2.imshow('mask', img_mask)

    img_height, img_width = frame.shape[:2]
    for i in range(0, img_height, grid_size):
        for j in range(0, img_width, grid_size):
            x_start, y_start = j, i
            x_end, y_end = x_start + grid_size, y_start + grid_size
            obstacle_found = False

            cell_mask = img_mask[y_start:y_end, x_start:x_end]
            if np.any(cell_mask):
                obstacle_found = True

            if obstacle_found:
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), -1)
            else:
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 0), 1)


def detect_thymio(frame):
    pass


def detect_goal(frame):
    pass


def main():
    # -------- Constants -------- #
    CAMERA_ID = 0

    MAP_MAX_HEIGHT = 600
    MAP_MAX_WIDTH = 800
    GRID_SIZE = 30 # Size of a grid cell in pixels
    # -------- Variables -------- #
    map_detection = False
    obstacles_detection = False

    map_coords = []
    obstacles_contours = []

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

            # Step 2: Detect the obstacles inside the map
            if obstacles_detection:
                obstacles_contours = detect_obstacles(map_frame)
                obstacles_detection = False
            
            if len(obstacles_contours) > 0:
                draw_obstacles(map_frame, obstacles_contours, GRID_SIZE, draw_contours=False)

            cv2.imshow('Map', map_frame)

        cv2.imshow('frame', frame_copy)

        # ---------- Keyboard options ---------- #
        # 1. Detect the map, obstacles and goal
        if cv2.waitKey(1) & 0xFF == ord('m'):
            print('Key m pressed')
            map_detection = True

        # 2. Path planning
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print('Key p pressed')

        # 3. Start the project
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print('Key s pressed')

        # 4. Quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Closing the program...')
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
