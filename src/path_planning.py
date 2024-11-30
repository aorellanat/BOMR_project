import numpy as np
import cv2
import time
from collections import deque

def compute_global_path(Nodes, Starting_node,Arrival_node, Mask):
    ###################################### import data from CV            #########################################################################################################################################
    Starting_node=0             ### index OF THE START AND GOAL
    Arrival_node=1
    Nodes=np.array([[97, 364],[636, 431], [504, 400], [529, 517], [294, 563], [271, 445], [579,  73], [678, 295], [560, 345], [464, 121], [356,  76], [384, 182], [156, 244], [128, 137]])
    Mask = cv2.imread("python_djikstra/mask_obstacles.png", cv2.IMREAD_COLOR)  ###### DATA FOR MASK (OBSTACLES)
    Threshold=1 ### potential use is we observe a mask that is uncorrect, with parasitic lines for ex.
    ###################################### end from CV        #####################################################################################################################################################

    
    ####################################### Start Djikstra - calculate connectivity and distances 
    Number_nodes = len(Nodes)
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
    while Visited_nodes[Arrival_node]!=True:
            Unvisited_nodes = np.where(Visited_nodes == False)[0]           ## unvisited indices
            Actual_node = Unvisited_nodes[np.argmin(Distance_from_start[Unvisited_nodes])]  ##minimal distance of the unvisited 
            for Neighbour_node in range(Number_nodes):   ##Comparaison avec d'autres noeuds accessibles
                    if Connectivity_matrix[min(Neighbour_node, Actual_node), max(Neighbour_node, Actual_node)] == True:
                            if (Distance_from_start[Neighbour_node]> Distance_from_start[Actual_node] + Distance_between_nodes[min(Neighbour_node,Actual_node),max(Neighbour_node,Actual_node)]):
                                    Distance_from_start[Neighbour_node]= Distance_from_start[Actual_node] + Distance_between_nodes[min(Neighbour_node,Actual_node),max(Neighbour_node,Actual_node)]         
                                    Previous[Neighbour_node]=Actual_node
            Visited_nodes[Actual_node] = True
    
            if (time.time() -Starting_time > 60):print("No path found within "+ str(round((time.time() -Starting_time),2)) +"seconds");break#### exit condition
    
    ######################################   GENERATE OUTPUT vector 
    Global_path = deque(Nodes[Arrival_node])
    Actual_node = Arrival_node
    while Actual_node!=Starting_node:                                                               ####looks at the end, what is previous node of actual, previous of previous etc....
        #Global_path = np.vstack([Global_path, Nodes[int(Previous[Actual_node])]]) 
        Global_path.appendleft(Nodes[Actual_node])
        Actual_node= int(Previous[Actual_node])
