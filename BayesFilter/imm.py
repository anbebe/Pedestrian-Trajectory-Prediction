import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.kalman import IMMEstimator
from bayes import Bayes

class IMM_CVCA(Bayes):
    """ 
    Interacting Multiple Models with Constant Velocity and constant acceleration
    """
    def __init__(self, name="IMM_CVCA"):
        super(IMM_CVCA, self).__init__(name=name)
        # set to default values
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 1, 'M': [[0.9, 0.1],[0.1, 0.9]], 'dt': 0.1, 'omega_variance': 0.1}
        
    def create_imm_cvca_estimator(self, initial_state):
        state_dim = 9  # Use a consistent state dimension
        measurement_dim = 3

        dt = self.params['dt']

        # Create a KalmanFilter instance for constant velocity
        kf_cv = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        kf_cv.F = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                            [0, 0, 0, 1, 0, 0, dt, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, dt, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        kf_cv.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        kf_cv.P *= 10
        kf_cv.Q = np.eye(state_dim) * 0.1
        kf_cv.R = np.eye(measurement_dim) * 1.0
        kf_cv.x = np.hstack((initial_state[:6], np.zeros(3)))  # Pad with zeros to match the state dimension

        # Create a KalmanFilter instance for constant acceleration
        kf_ca = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        kf_ca.F = np.array([[1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
                            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
                            [0, 0, 0, 1, 0, 0, dt, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, dt, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, dt],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        kf_ca.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0]])
        kf_ca.P *= 10
        kf_ca.Q = np.eye(state_dim) * 0.1
        kf_ca.R = np.eye(measurement_dim) * 1.0
        kf_ca.x = initial_state

        # Define initial mode probabilities
        mu = np.array([0.5, 0.5])

        # Define Markov chain transition matrix
        M = np.asarray(self.params['M'])

        # Create IMM Estimator
        return IMMEstimator([kf_cv, kf_ca], mu, M)

    def predict(self, batch_positions):

        batch_size, timesteps, _ = batch_positions.shape

        dt = self.params['dt']
        
        predictions = np.zeros((batch_size, 15, 9))
        
        for i in range(batch_size):
            # Initial state (starting with the first position, estimated velocity, and zero acceleration)
            initial_position = batch_positions[i, 0]
            initial_velocity = (batch_positions[i, 1] - batch_positions[i, 0]) / dt
            initial_acceleration = np.zeros(3)
            initial_state = np.hstack((initial_position, initial_velocity, initial_acceleration))

            imm = self.create_imm_cvca_estimator(initial_state)
            
            # Running the IMM estimator for the first 5 timesteps
            for t in range(5):
                z = batch_positions[i, t]
                imm.predict()
                imm.update(z)
                predictions[i, t] = imm.x
            
            # Predicting the next 10 timesteps without updating
            for t in range(5, 15):
                imm.predict()
                predictions[i, t] = imm.x

        return predictions


class IMM_CVCT(Bayes):
    """ 
    Interacting Multiple Models with Constant Velocity and constant acceleration
    """
    def __init__(self, name="IMM_CVCT"):
        super(IMM_CVCT, self).__init__(name=name)
        # set to default values
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 1, 'M': [[0.85, 0.15],  # High likelihood of staying in CV
                [0.15, 0.85]], 'dt': 0.1, 'omega_variance': 0.1}

    def create_imm_cvct_estimator(self, initial_state):
        state_dim = 10  # Consistent state dimension across both models
        measurement_dim = 3

        dt = self.params['dt']
        q = self.params['q']
        r = self.params['r']
        omega_variance = self.params['omega_variance']

        # Create a KalmanFilter instance for Constant Velocity (CV) Model
        kf_cv = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        # State Transition Matrix for CV
        kf_cv.F = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, dt, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, dt, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, dt, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Zero acceleration
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Zero acceleration
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Zero acceleration
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) # Turn rate constant
        
        kf_cv.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        kf_cv.P = np.eye(state_dim) * 10
        kf_cv.Q = np.eye(state_dim) * q
        kf_cv.R = np.eye(measurement_dim) * r
        kf_cv.x = initial_state.copy()
        kf_cv.x[9] = 0  # Zero turn rate for CV model

        # Create a KalmanFilter instance for Coordinated Turn (CT) Model
        # Create a KalmanFilter instance for Coordinated Turn (CT) Model
        kf_ct = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        
        # Initial turn rate (omega)
        omega = initial_state[9]  # Turn rate
        
        # Avoid division by zero for small omega
        if omega == 0:
            omega = 1e-5
        
        sin_omega_dt = np.sin(omega * dt)
        cos_omega_dt = np.cos(omega * dt)
        
        # State Transition Matrix for CT
        kf_ct.F = np.array([
            [1, 0, 0, sin_omega_dt/omega, 0, 0, (1 - cos_omega_dt)/omega, 0, 0, 0],
            [0, 1, 0, 0, sin_omega_dt/omega, 0, 0, (1 - cos_omega_dt)/omega, 0, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0, 0],
            [0, 0, 0, cos_omega_dt, 0, 0, sin_omega_dt, 0, 0, 0],
            [0, 0, 0, 0, cos_omega_dt, 0, 0, sin_omega_dt, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Zero acceleration in x
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Zero acceleration in y
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Zero acceleration in z
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # Turn rate remains constant
        ])

        
        kf_ct.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        kf_ct.P = np.eye(state_dim) * 10
        kf_ct.Q = np.eye(state_dim) * q
        # Specifically, for the turn rate component
        # Assuming omega_variance is an additional hyperparameter
        kf_ct.Q[6, 6] = q * omega_variance
        kf_ct.Q[7, 7] = q * omega_variance
        kf_ct.Q[9, 9] = q * omega_variance  # Higher uncertainty in the turn rate

        kf_ct.R = np.eye(measurement_dim) * r
        kf_ct.x = initial_state.copy()

        # Define initial mode probabilities
        mu = np.array([0.5, 0.5])

        # Define Markov chain transition matrix
        M = np.array(self.params['M'])  # High likelihood of staying in CT


        # Create IMM Estimator
        return IMMEstimator([kf_cv, kf_ct], mu, M)

    def predict(self, batch_positions):
        batch_size, timesteps, _ = batch_positions.shape
        
        predictions = np.zeros((batch_size, 15, 10))  # Assuming 10 state dimensions
        dt = self.params['dt']
        q = self.params['q']
        r = self.params['r']
        omega_variance = self.params['omega_variance']
        
        for i in range(batch_size):
            # Initial state estimation
            initial_position = batch_positions[i, 0]
            initial_velocity = (batch_positions[i, 1] - batch_positions[i, 0]) / dt
            initial_acceleration = np.zeros(2)
            initial_turn_rate = 0.0  # Assuming starting with no turn
            initial_state = np.hstack((initial_position, initial_velocity, initial_turn_rate, initial_acceleration))
            initial_state = np.hstack((initial_state, 0))  # Padding to match state dimension if necessary

            # Create IMM estimator with the provided hyperparameters
            imm = self.create_imm_cvct_estimator(initial_state)
            
            # Running the IMM estimator for the first 5 timesteps
            for t in range(5):
                z = batch_positions[i, t]
                imm.predict()
                imm.update(z)
                predictions[i, t] = imm.x
            
            # Predicting the next 10 timesteps without updating (no measurements)
            for t in range(5, 15):
                imm.predict()
                predictions[i, t] = imm.x

        return predictions



