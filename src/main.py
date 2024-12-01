from cv import *
from path_planning import *

# -------- Computer vision constants -------- #
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
    mask_obstacles = None

    thymio_found = False
    thymio_coords = []
    thymio_angle = None
    goal_coords = None

    path = []

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
                path = compute_global_path(thymio_coords, goal_coords, obstacle_vertices, mask_obstacles)
            else:
                print(f'No path found. Thymio: {thymio_found}, Goal: {goal_coords}')
            path_planning = False


        # Step 3: Start motion
        if start_motion:
            if len(path) > 0:
                thymio_coords_cm = convert_pixel_to_cm(thymio_coords, REAL_MAP_WIDTH_CM, REAL_MAP_HEIGHT_CM, MAP_MAX_WIDTH, MAP_MAX_HEIGHT)
                # ------> Important: Here you have the position, and angle of the thymio
            else:
                print('No path to follow')


        # ---------- Display the frames ---------- #
        if len(obstacles_contours) > 0:
            draw_obstacles(map_frame, obstacles_contours)

        if goal_coords:
            draw_goal(map_frame, goal_coords)

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
            start_motion = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Closing the program...')
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
