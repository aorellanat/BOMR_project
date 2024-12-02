import numpy as np
from utils import *
import matplotlib.pyplot as plt
NOISE_COV_VW = np.array([[0.01285577, 0.00072932],
                   [0.00072932, 0.00056978]]) # from calibration.ipynb
NOISE_COV_CAMERA = 0.0001*np.eye(3)
NOISE_COV_CAMERA_BLOCKED=9999999*np.eye(3)
PROCESS_COV = 0.01*np.eye(5)
class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, control_dim, dt):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.control_dim = control_dim
        self.dt = dt
        self.P = 0.1*np.eye(self.state_dim)
        self.H = np.eye(self.measurement_dim)
        self.state_trajectory = []
        self.camera_available = True # initially, assumes that camera is available and sets noise variance accordingly

    def initialize_X(self, initial_X):
        assert(initial_X.shape[0] == self.state_dim)
        self.X = initial_X
        self.X[2] = self.clip_theta(self.X[2])
        self.Q = PROCESS_COV # process_noise
        self.R = np.block([[NOISE_COV_CAMERA, np.zeros((3,2))],[np.zeros((2,3)), NOISE_COV_VW]]) # measurement_noise
        assert(self.Q.shape[0] == self.state_dim)
        assert(self.R.shape[0] == self.measurement_dim)
        self.state_trajectory.append(initial_X)

    # takes a boolean value camera_blocked and modifies noise variance matrix
    def switch_mode(self, camera_blocked):
        if camera_blocked and self.camera_available:
            self.R[0:3, 0:3] = NOISE_COV_CAMERA_BLOCKED
            self.camera_available = False
        elif not camera_blocked and not self.camera_available:
            self.R[0:3,0:3] = NOISE_COV_CAMERA
            self.camera_available = True

    def move(self, u):
        assert(u.shape[0] == 2) # u is np.array containing v_,w_
        x,y,theta,v,w = self.X
        v_, w_ = u
        self.X[0] = x + v*np.cos(theta)*self.dt 
        self.X[1] = y + v*np.sin(theta)*self.dt
        self.X[2] = self.clip_theta(theta + w * self.dt)
        self.X[3] = v_ 
        self.X[4] = w_

    def compute_F(self):
        x,y,theta,v,w = self.X
        F = np.array([[1,0,-v*np.sin(theta)*self.dt, np.cos(theta)*self.dt, 0],
                      [0,1,v*np.cos(theta)*self.dt,np.sin(theta)*self.dt, 0],
                      [0,0,1,0,self.dt],
                      [0,0,0,0,0],
                      [0,0,0,0,0]])
        
        return F
    
    # u = np.array([v_, w_])
    # z = np.array([x,y,theta,v,w])
    def predict_and_update(self, u, z):
        assert(u.shape[0] == self.control_dim)
        assert(z.shape[0] == self.measurement_dim)
        F = self.compute_F()
        self.move(u)
        self.P = F @ self.P @ F.T + self.Q
        y = z - self.X # use measurement model H(X) = X
        y[2] = self.clip_theta(y[2])
        S = self.H @ self.P @ self.H.T + self.R
        S_inv = np.linalg.inv(S)
        K = self.P @ self.H.T @ S_inv
        self.X = self.X + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
        self.state_trajectory.append(self.X)
    # theta (or delta theta) must be in the range of [-pi, pi)

    def get_X(self):
        return self.X
    
    def clip_theta(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi


