import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
import matplotlib.pyplot as plt
from bayes import Bayes
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from sympy.abc import alpha, x, y, v, w, R, theta
import sympy as sp
from filterpy.kalman import ExtendedKalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
from sklearn.model_selection import ParameterGrid


class Kalman_CV(Bayes):
    """
    Class for a Kalman Filter with a constant velocity model
    """

    def __init__(self, pos_dim, name="Kalman_CV"):
        super(Kalman_CV, self).__init__(pos_dim=pos_dim, name=name)
        # set to default values
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 0.1, 'dt': 0.4}

    def predict(self, batch_positions):
        """
        Applies Kalman Filter to the first 5 timesteps to predict the next 10 timesteps based on the 
        constant velocity model. Can be applied to 3D and 2D coordinates.
        
        :param batch_positions: A numpy array of shape (batch_size, 15, 3) containing the  positions.
                            
        :returns  numpy array of shape (batch_size, 15, 6) containing the predicted states (position and velocity)
        for each timestep in each batch.
        """
        batch_size, timesteps, _ = batch_positions.shape
        tmp_t = 8
        predictions = np.zeros((batch_size, 15, self.pos_dim*2))
        dt = self.params['dt']

        for i in range(batch_size):
            # create KF and CV model for 3D positions
            if self.pos_dim == 3:
                kf = KalmanFilter(dim_x=6, dim_z=3)
                # State Transition matrix A
                kf.F = np.array([[1, 0, 0, dt, 0, 0],
                                [0, 1, 0, 0, dt, 0],
                                [0, 0, 1, 0, 0, dt],
                                [0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
                # Measurement matrix H
                kf.H = np.array([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0]])
                # Initial state (starting with the first position and zero velocity)
                initial_position = batch_positions[i, 0]
                kf.x = np.array([initial_position[0], initial_position[1], initial_position[2], 0, 0, 0])
            # create KF and CV model for 2D positions
            elif self.pos_dim == 2:
                kf = KalmanFilter(dim_x=4, dim_z=2)
                # State Transition matrix A
                kf.F = np.array([[1, 0, dt, 0],
                                [0, 1, 0, dt],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
                # Measurement matrix H
                kf.H = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]])
                # Initial state (starting with the first position and zero velocity)
                initial_position = batch_positions[i, 0]
                kf.x = np.array([initial_position[0], initial_position[1], 0, 0])
            # The following params are independent of the dimension of the position
            # Initial state covariance
            kf.P *= self.params['P']
            # Process noise covariance
            q = Q_discrete_white_noise(dim=self.pos_dim, dt=dt, var=0.001)
            kf.Q = block_diag(q, q)
            # Measurement noise covariance
            kf.R = np.eye(self.pos_dim) * 0.1
            
            
            # Run the Kalman Filter for the first 5 timesteps
            for t in range(tmp_t):
                z = batch_positions[i, t]
                kf.predict()
                kf.update(z)
                predictions[i, t] = kf.x
            
            # Predict the next 10 timesteps without updating
            for t in range(tmp_t, 15):
                kf.predict()
                predictions[i, t] = kf.x
        
        return predictions
    
    def hyperparameter_tuning(self, batch_positions):
        """
            Applies hyperparameter tuning by trying the KF with different sets of parameters 
            and setting the internal parameters to the set achieving the best ADE.

            :param batch_positions tarjectories with the shape (batch_size, sequence_length, 2)
        """
        # Define the hyperparameter space
        param_grid = {
            'q': [0.1, 0.2, 0.5, 0.8], 
            'r': [0.1, 0.2, 0.5, 0.8], 
            'P':[0.1, 0.5, 1.0], 
            'dt':  [0.1,0.4,0.9]        }

        grid = ParameterGrid(param_grid)
        best_score = float('inf')
        best_params = None

        # Loop through all combinations of hyperparameters
        for params in grid:
            self.params['q'] = params['q']
            self.params['r'] = params['r']
            self.params['P'] = params['P']
            self.params['dt'] = params['dt']
            
            predictions = self.predict(batch_positions)
            error = self.calculate_meanADE(batch_positions, predictions)

            # Update the best parameters
            if error < best_score:
                best_score = error
                best_params = params

        print("Best Hyperparameters:", best_params)
        print("Best score:", best_score)
        self.hyperparameters = best_params
    
class Kalman_CA(Bayes):
    """
    Class for a Kalman Filter with a constant acceleration model
    """

    def __init__(self, pos_dim, name="Kalman_CA"):
        super(Kalman_CA, self).__init__(pos_dim=pos_dim, name=name)
        # set to default values
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 1, 'dt': 0.4}

    def predict(self, batch_positions):
        """
        Applies Kalman Filter to the first 5 timesteps to predict the next 10 timesteps based on the 
        constant velocity model. Can be applied to 3D and 2D coordinates.
        
        :param batch_positions: A numpy array of shape (batch_size, 15, 3) containing the  positions.
                            
        :returns  numpy array of shape (batch_size, 15, 6) containing the predicted states (position and velocity)
        for each timestep in each batch.
        """
        batch_size, timesteps, _ = batch_positions.shape
        
        predictions = np.zeros((batch_size, 15, self.pos_dim*3))

        dt = self.params['dt']
        
        for i in range(batch_size):
            kf = KalmanFilter(dim_x=self.pos_dim*3, dim_z=self.pos_dim)
            if self.pos_dim == 3:            
                # State Transition matrix A
                kf.F = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                                [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                                [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 1]])
                # Measurement matrix H
                kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 0]])
            elif self.pos_dim == 2:
                # State Transition matrix A 
                kf.F = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                                [0, 1, 0, dt, 0, 0.5*dt**2],
                                [0, 0, 1, 0, dt, 0],
                                [0, 0, 0, 1, 0, dt],
                                [0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 1]])
                # Measurement matrix H
                kf.H = np.array([[1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0]])
            # Initial state covariance
            kf.P *= self.params['P']
            # Process noise covariance
            kf.Q = np.eye(self.pos_dim*3) * 0.01
            # Measurement noise covariance
            kf.R = np.eye(self.pos_dim) * 0.1
            # Initial state (starting with the first position, estimated velocity, and zero acceleration)
            initial_position = batch_positions[i, 0]
            initial_velocity = (batch_positions[i, 1] - batch_positions[i, 0]) / dt
            initial_acceleration = np.zeros(self.pos_dim)
            kf.x = np.hstack((initial_position, initial_velocity, initial_acceleration))
            
            # Running the Kalman Filter for the first 5 timesteps
            for t in range(5):
                z = batch_positions[i, t]
                kf.predict()
                kf.update(z)
                predictions[i, t] = kf.x
            
            # Predicting the next 10 timesteps without updating
            for t in range(5, 15):
                kf.predict()
                predictions[i, t] = kf.x
        
        return predictions



  
