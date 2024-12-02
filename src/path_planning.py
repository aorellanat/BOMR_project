import numpy as np
import cv2
import time
from collections import deque

TIMEOUT = 10

def compute_global_path(Starting_node_pos, Arrival_node_pos, Nodes, Mask, Threshold=1, visibility_grap_width=500, visibility_grap_height=400):
    ###################################### import data from CV            #########################################################################################################################################
    Starting_node_pos=np.array(Starting_node_pos, dtype=np.int32)
    Arrival_node_pos=np.array(Arrival_node_pos, dtype=np.int32)
    Nodes.insert(0, Starting_node_pos)
    Nodes.insert(1, Arrival_node_pos)

    Starting_node = 1            ### index OF THE START AND GOAL
    Arrival_node = 0
    ###################################### end from CV        #####################################################################################################################################################
    
    ####################################### Start Djikstra - calculate connectivity and distances 
    Number_nodes = len(Nodes)

    print(Nodes)
    Connectivity_matrix = np.zeros((Number_nodes, Number_nodes), dtype=bool)      ##Connectivity_matrix[i, j] = 1 means i connected to j !!!!!!!!!!!!!!!!! i<j !!!!!!!!!! Matrix verifying if i and j are connected. 1 ou 0 (triangulaire supperieure)
    Distance_between_nodes = np.full((Number_nodes, Number_nodes), np.inf)  #Initialise distances at infinity, to be changed.Also a matrice that is triangular supperior. The diagonal should be 0, but since we don't use it we don't lose time doing it
    Starting_time=time.time();Exit= bool(0)
    
    for j in range(Number_nodes):
        for i in range(j):        #Connectivity_matrix[i, j] = 1 means i connected to j !!!!!!!!!!!!!!!!! i<j !!!!!!!!!!    i in range(j) compared to i in range(Number_nodes) increases the speed by two.
            Blank_img=np.zeros(shape=Mask.shape, dtype=np.uint8)   #(https://medium.com/jungletronics/opencv-image-basics-2e63d973851a) # Blank image for addinf segment
            cv2.line(Blank_img,Nodes[i],Nodes[j],(127,127,127),1)
            if(np.sum(cv2.bitwise_and(Blank_img, Mask))<=Threshold):
                Connectivity_matrix[i,j]=True
                Distance_between_nodes[i,j] = np.linalg.norm(Nodes[i]-Nodes[j])    ## l2-norm for distance
    
    ######################################## Backtracking Initialisation   (https://www.freecodecamp.org/news/dijkstras-algorithm-explained-with-a-pseudocode-example/)
    Visited_nodes = np.zeros(Number_nodes, dtype=bool)         ## what nodes are already treated
    Distance_from_start = np.full(Number_nodes, np.inf)     ####calculate distance fromm start for each node
    Distance_from_start[Starting_node]=0                    ### set the starting node to 0
    Previous = np.full((Number_nodes,), np.nan)   ## for each node know the previous node
    Previous[Starting_node]=Starting_node
    
    #######################################  Backtracking    
    Unvisited_nodes = np.where(Visited_nodes == False)[0]
    # print(Connectivity_matrix[0, :])

    while Visited_nodes[Arrival_node]!=True:
            Unvisited_nodes = np.where(Visited_nodes == False)[0]           ## unvisited indices
            Actual_node = Unvisited_nodes[np.argmin(Distance_from_start[Unvisited_nodes])]  ##minimal distance of the unvisited 
            for Neighbour_node in range(Number_nodes):   ##Comparaison avec d'autres noeuds accessibles
                    if Connectivity_matrix[min(Neighbour_node, Actual_node), max(Neighbour_node, Actual_node)] == True:
                            if (Distance_from_start[Neighbour_node]> Distance_from_start[Actual_node] + Distance_between_nodes[min(Neighbour_node,Actual_node),max(Neighbour_node,Actual_node)]):
                                    Distance_from_start[Neighbour_node]= Distance_from_start[Actual_node] + Distance_between_nodes[min(Neighbour_node,Actual_node),max(Neighbour_node,Actual_node)]         
                                    Previous[Neighbour_node]=Actual_node
            Visited_nodes[Actual_node] = True
    
            if (time.time() -Starting_time > TIMEOUT):print("No path found within "+ str(round((time.time() -Starting_time),2)) +"seconds");break#### exit condition
    
    ######################################   GENERATE OUTPUT vector 
    Global_path = deque(Nodes[Arrival_node])
    Actual_node = Arrival_node
    while Actual_node!=Starting_node:                                                               ####looks at the end, what is previous node of actual, previous of previous etc....
        Global_path = np.vstack([Global_path, Nodes[int(Previous[Actual_node])]]) 
        #Global_path.appendleft(Nodes[Actual_node])
        Actual_node= int(Previous[Actual_node])

    draw_visibility_graph(
        Mask,
        Nodes,
        Connectivity_matrix,
        Starting_node,
        Arrival_node,
        Previous,
        visibility_grap_width,
        visibility_grap_height
    )

    print(f"Global path: {Global_path}")
    return Global_path


def draw_path(path, image):
    for i in range(len(path) - 1):
        cv2.line(image, tuple(path[i]), tuple(path[i+1]), (0, 255, 0), 2)


def draw_visibility_graph(Mask, Nodes, Connectivity_matrix, Starting_node, Arrival_node, Previous, width, height):
    visibility_graph = Mask.copy()
    Nodes = np.array(Nodes)

    # Draw connections from the Connectivity_matrix
    for j in range(len(Nodes)):
        for i in range(j):
            if Connectivity_matrix[i, j] == 1:  # i < j
                start = (int(Nodes[i, 0]), int(Nodes[i, 1]))
                end = (int(Nodes[j, 0]), int(Nodes[j, 1]))
                cv2.line(visibility_graph, start, end, (255, 0, 0), 1, lineType=cv2.LINE_AA)  # Blue dashed line alternative

    # Draw nodes and labels
    for i in range(len(Nodes)):
        center = (int(Nodes[i, 0]), int(Nodes[i, 1]))
        cv2.circle(visibility_graph, center, 5, (255, 0, 255), -1)  # Magenta nodes
        label_pos = (center[0] + 5, center[1] - 5)
        cv2.putText(visibility_graph, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1, lineType=cv2.LINE_AA)

    # Draw the actual path
    Actual_node = Arrival_node
    while Actual_node != Starting_node:
        start = (int(Nodes[Actual_node, 0]), int(Nodes[Actual_node, 1]))
        end = (int(Nodes[int(Previous[Actual_node]), 0]), int(Nodes[int(Previous[Actual_node]), 1]))
        cv2.line(visibility_graph, start, end, (0, 0, 255), 2, lineType=cv2.LINE_AA)  # Red line
        Actual_node = int(Previous[Actual_node])

    # Highlight start and goal nodes
    cv2.circle(visibility_graph, (int(Nodes[Starting_node, 0]), int(Nodes[Starting_node, 1])), 10, (255, 0, 0), -1)  # Blue start
    cv2.circle(visibility_graph, (int(Nodes[Arrival_node, 0]), int(Nodes[Arrival_node, 1])), 10, (0, 0, 255), -1)  # Red goal

    visibility_graph = cv2.resize(visibility_graph, (width, height))
    cv2.imshow("Visibility graph", visibility_graph)
