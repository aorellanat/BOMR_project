import numpy as np
from utils import *
import matplotlib.pyplot as plt

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim, control_dim, dt):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.control_dim = control_dim
        self.dt = dt
        self.P = np.eye(self.state_dim)
        self.H = np.eye(self.measurement_dim)
        self.state_trajectory = []

    def initialize_X_Q_R(self, initial_X, initial_Q, initial_R):
        assert(initial_X.shape[0] == self.state_dim)
        assert(initial_Q.shape[0] == self.state_dim)
        assert(initial_R.shape[0] == self.measurement_dim)
        self.X = initial_X
        self.X[2] = self.clip_theta(self.X[2])
        self.Q = initial_Q # process_noise
        self.R = initial_R # measurement_noise
        self.state_trajectory.append(initial_X)
    
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
                      [0,0,0,1,0],
                      [0,0,0,0,1]])
        
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
    def clip_theta(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi


