import numpy as np
wheel_axle_length = 9.5 # in cm
thymio_speed_to_cms = 0.03260
A_theory = thymio_speed_to_cms* np.array([[0.5, 0.5],[-1/wheel_axle_length, 1/wheel_axle_length]]) # theoretical kinematics model, but is overestimating
tuning_matrix = np.array([[1.65,1.65],[2,2]])# compensate for overestimation
A_tuned = np.multiply(tuning_matrix, A_theory)

def from_u_to_vw(ul, ur):
    vw = A_tuned@np.array([ul, ur]) 
    return vw[0], vw[1] # returns v, w in cm/s, rad/s

def from_vw_to_u(v,w):
    A_inv = np.linalg.inv(A_tuned)
    ulur = A_inv @ np.array([v,w]) 
    return int(ulur[0]), int(ulur[1]) # returns ul, ur as int
