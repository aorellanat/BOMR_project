import threading
import time

from cv import *
from kalman import *

# -------- Constants -------- #
CAMERA_ID = 0

REAL_MAP_HEIGHT_CM = 84
REAL_MAP_WIDTH_CM = 88.5

MAP_MAX_HEIGHT = 600
MAP_MAX_WIDTH = 800

PADDING_OBSTACLES = 30

def main():
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

    camera = cv2.VideoCapture(CAMERA_ID)
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
            pts2 = np.float32([[0, 0], [MAP_MAX_WIDTH, 0], [MAP_MAX_WIDTH, MAP_MAX_HEIGHT], [0, MAP_MAX_HEIGHT]])
            matrix = cv2.getPerspectiveTransform(map_coords, pts2)
            map_frame = cv2.warpPerspective(frame, matrix, (MAP_MAX_WIDTH, MAP_MAX_HEIGHT))

            # Step 2: Detect the obstacles inside the map and the goal
            if obstacles_detection:
                obstacles_contours, mask_obstacles, goal_coords = detect_obstacles_and_goal(map_frame, PADDING_OBSTACLES)
                obstacles_detection = False
            
            if len(obstacles_contours) > 0:
                draw_obstacles(map_frame, obstacles_contours)

            if goal_coords:
                draw_goal(map_frame, goal_coords)

            thymio_found, thymio_coords, thymio_angle = detect_thymio(map_frame)

            # Step 3: Path planning
            if path_planning:
                if thymio_found and goal_coords:
                    obstacle_vertices = get_obstacle_vertices(obstacles_contours)
                    path = compute_global_path(thymio_coords, goal_coords, obstacle_vertices, mask_obstacles)
                else:
                    print(f'It was not possible to detect the path planning. Thymio: {thymio_found}, Goal: {goal_coords}')
                path_planning = False

            # Step 4: Init the project
            if start_motion:
                print('Starting the project...')
                thymio_coords_cm = convert_pixel_to_cm(thymio_coords, REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                print(f'Thymio coordinates in cm: {thymio_coords_cm}')
                # ------> Important: Here you have the position, and angle of the thymio


            if len(path) > 0:
                draw_path(path, map_frame)

            # Reshape map before display it
            map_frame = cv2.resize(map_frame, (500, 400))
            cv2.imshow('Map', map_frame)

        if thymio_found:
            cv2.putText(frame, f'Thymio (x,y): {thymio_coords}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 3)
            cv2.putText(frame, f'Thymio angle rad: {thymio_angle:.4f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 3)
            cv2.putText(frame, f'Thymio angle deg: {np.degrees(thymio_angle):.4f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 3)

        cv2.imshow('frame', frame)

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

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
