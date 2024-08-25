import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from bayes import Bayes
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise

class Kalman_CV(Bayes):
    """
    Constant Velocity Kalman Filter
    """

    def __init__(self, name="Kalman_CV"):
        super(Kalman_CV, self).__init__(name=name)
        # set to default values
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 0.1, 'dt': 0.4}

    def predict(self, batch_positions):
        """
        Constant Velocity 

        Applies Kalman Filter to the first 5 timesteps to predict the next 10 timesteps.
        
        Args:
            batch_positions: A numpy array of shape (batch_size, 15, 3)
                            containing the 3D positions.
                            
        Returns:
            A numpy array of shape (batch_size, 15, 6) containing the
            predicted states (position and velocity) for each timestep in each batch.
        """
        batch_size, timesteps, _ = batch_positions.shape
        assert timesteps == 15, "The input should have 15 timesteps per sequence."
        
        predictions = np.zeros((batch_size, 15, 6))
        dt = self.params['dt']

        for i in range(batch_size):
            
            # Create a KalmanFilter instance
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
            
            # Initial state covariance
            kf.P *= self.params['P']
            
            # Process noise covariance
            q = Q_discrete_white_noise(dim=3, dt=dt, var=0.001)
            kf.Q = block_diag(q, q)
            
            # Measurement noise covariance
            kf.R = np.eye(3) * 0.1
            
            # Initial state (starting with the first position and zero velocity)
            initial_position = batch_positions[i, 0]
            kf.x = np.array([initial_position[0], initial_position[1], initial_position[2], 0, 0, 0])
            
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
    

class Kalman_CA(Bayes):
    """
    Constant Acceleration Kalman Filter
    """

    def __init__(self, name="Kalman_CA"):
        super(Kalman_CA, self).__init__(name=name)
        # set to default values
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 1, 'dt': 0.4}

    def predict(self, batch_positions):
        """
        Constant acceleration

        Applies Kalman Filter to the first 5 timesteps to predict the next 10 timesteps.
        
        Args:
            batch_positions: A numpy array of shape (batch_size, 15, 3)
                            containing the 3D positions.
                            
        Returns:
            A numpy array of shape (batch_size, 15, 9) containing the
            predicted states (position, velocity, and acceleration) for each timestep in each batch.
        """
        batch_size, timesteps, _ = batch_positions.shape
        assert timesteps == 15, "The input should have 15 timesteps per sequence."
        
        predictions = np.zeros((batch_size, 15, 9))

        dt = self.params['dt']
        
        for i in range(batch_size):
            
            # Create a KalmanFilter instance
            kf = KalmanFilter(dim_x=9, dim_z=3)
            
            # State Transition matrix A (constant acceleration model)
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
            
            # Initial state covariance
            kf.P *= self.params['P']
            
            # Process noise covariance
            kf.Q = np.eye(9) * 0.01
            
            # Measurement noise covariance
            kf.R = np.eye(3) * 0.1
            
            # Initial state (starting with the first position, estimated velocity, and zero acceleration)
            initial_position = batch_positions[i, 0]
            initial_velocity = (batch_positions[i, 1] - batch_positions[i, 0]) / dt
            initial_acceleration = np.zeros(3)
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