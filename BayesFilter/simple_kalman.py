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


class Kalman_CV(Bayes):
    """
    Constant Velocity Kalman Filter
    """

    def __init__(self, pos_dim, name="Kalman_CV"):
        super(Kalman_CV, self).__init__(pos_dim=pos_dim, name=name)
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
        
        predictions = np.zeros((batch_size, 15, self.pos_dim*2))
        dt = self.params['dt']

        for i in range(batch_size):
            
            if self.pos_dim == 3:
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
                
                # Initial state (starting with the first position and zero velocity)
                initial_position = batch_positions[i, 0]
                kf.x = np.array([initial_position[0], initial_position[1], initial_position[2], 0, 0, 0])
            
            elif self.pos_dim == 2:
                # Create a KalmanFilter instance
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
            
            # Initial state covariance
            kf.P *= self.params['P']
            
            # Process noise covariance
            q = Q_discrete_white_noise(dim=self.pos_dim, dt=dt, var=0.001)
            kf.Q = block_diag(q, q)
            
            # Measurement noise covariance
            kf.R = np.eye(self.pos_dim) * 0.1
            
            
            
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

    def __init__(self, pos_dim, name="Kalman_CA"):
        super(Kalman_CA, self).__init__(pos_dim=pos_dim, name=name)
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
        
        predictions = np.zeros((batch_size, 15, self.pos_dim*3))

        dt = self.params['dt']
        
        for i in range(batch_size):
            
            # Create a KalmanFilter instance
            kf = KalmanFilter(dim_x=self.pos_dim*3, dim_z=self.pos_dim)
            if self.pos_dim == 3:            
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
                
            elif self.pos_dim == 2:
                # State Transition matrix A (constant acceleration model)
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
    
class EKF_CT2(Bayes):
    """
    Extended Kalman Filter with constant turn rate
    """

    def __init__(self, pos_dim, name="Kalman_CV"):
        super(EKF_CT2, self).__init__(pos_dim=pos_dim, name=name)
        # set to default values
        if self.params is None:
            self.params = {'q': 0.001, 'r': 0.01, 'P': 0.1, 'dt': 0.1, 'omega': 0.02}

        # Define symbolic variables for sympy
        self.x1, self.x2, self.v1, self.v2, dt, omega = sp.symbols('x1 x2 v1 v2 dt omega')

        # Define the state transition matrix using sympy
        self.F_sym = sp.Matrix([
            [1, 0,sp.sin(omega * dt) / omega, -(1 - sp.cos(omega * dt)) / omega],
            [0, 0, sp.cos(omega * dt), -sp.sin(omega * dt)],
            [0, 1, 1 - sp.cos(omega * dt), sp.sin(omega * dt) / omega],
            [0, 0, sp.sin(omega * dt), sp.cos(omega * dt)]
        ])

        # State vector
        self.state_vector = sp.Matrix([self.x1, self.x2, self.v1, self.v2])

    def state_transition(self, x, dt, omega):
        """
        State transition function for the Constant Turn model.
        """
        F_CT = np.array([[1, 0, np.sin(omega * dt) / omega, -(1 - np.cos(omega * dt)) / omega],
                         [0, 0,np.cos(omega * dt), -np.sin(omega * dt)],
                         [0, 1, 1 - np.cos(omega * dt), np.sin(omega * dt) / omega],
                         [0, 0, np.sin(omega * dt), np.cos(omega * dt)]])
        return F_CT @ x

    def jacobian(self, x, dt, omega):
        """
        Jacobian of the state transition function with respect to the state.
        """
        # Substitute the current state values into the symbolic matrix
        subs = {self.x1: x[0], self.x2: x[1], self.v1: x[2], self.v2: x[3], 'dt': dt, 'omega': omega}
        F_J = np.array(self.F_sym.subs(subs)).astype(float)
        return F_J

    def hx(self, x):
        """
        Measurement function that maps the state vector into the measurement space.
        """
        return x[:2]

    def predict(self, batch_positions):
        """
        Constant Turn EKF Prediction using FilterPy.
        """
        batch_size, timesteps, _ = batch_positions.shape
        assert timesteps == 15, "The input should have 15 timesteps per sequence."
        
        predictions = np.zeros((batch_size, 15, self.pos_dim * 2))
        dt = self.params['dt']
        omega = self.params['omega']

        for i in range(batch_size):
            # Initialize the Kalman Filter
            kf = ExtendedKalmanFilter(dim_x=4, dim_z=2)

            # Measurement matrix H
            kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

            # Initial state (position and velocity)
            initial_position = batch_positions[i, 0]
            kf.x = np.array([initial_position[0], initial_position[1], 0, 0])

            # Initial state covariance
            kf.P *= self.params['P']

            # Process noise covariance
            q = Q_discrete_white_noise(dim=self.pos_dim, dt=dt, var=self.params['q'])
            kf.Q = block_diag(q, q)

            # Measurement noise covariance
            kf.R = np.eye(self.pos_dim) * self.params['r']

            # EKF prediction and update for the first 5 timesteps
            for t in range(5):
                z = batch_positions[i, t]

                # Update Jacobian matrix F based on the current state
                kf.F = self.jacobian(kf.x, dt, omega)

                # Perform the predict and update step
                kf.predict_update(z, HJacobian=lambda x, *args: kf.H, Hx=self.hx, args=(dt, omega), hx_args=())

                predictions[i, t] = kf.x

            # Predicting the next 10 timesteps without measurement update
            for t in range(5, 15):
                kf.F = self.jacobian(kf.x, dt, omega)
                kf.predict()
                predictions[i, t] = kf.x


        
        return predictions


  
