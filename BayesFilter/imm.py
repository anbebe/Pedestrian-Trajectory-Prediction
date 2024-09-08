import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.kalman import IMMEstimator
from bayes import Bayes
from sklearn.model_selection import ParameterGrid



class IMM_CVCT_3D(Bayes):
    """ 
    Interacting Multiple Models with Constant Velocity and constant turn
    """
    def __init__(self, pos_dim, name="IMM_CVCT"):
        super(IMM_CVCT_3D, self).__init__(pos_dim=pos_dim, name=name)
        # set to default values
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 1, 
                       'M': [[0.80, 0.05, 0.05, 0.05],  # High likelihood of staying in CV
                             [0.05, 0.8, 0.05, 0.05],
                             [0.05, 0.05, 0.8, 0.05],
                             [0.05, 0.05, 0.05, 0.8]], 
                       'dt': 0.4, 
                       'omega_variance': 0.1}

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
                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Zero acceleration
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Zero acceleration
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Zero acceleration
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]) # Turn rate constant
        
        kf_cv.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        kf_cv.P = np.eye(state_dim) * self.params['P']
        kf_cv.Q = np.eye(state_dim) * self.params['q']
        kf_cv.R = np.eye(measurement_dim) * self.params['r']
        kf_cv.x = initial_state.copy()
        kf_cv.x[9] = 0  # Zero turn rate for CV model

        # Create a KalmanFilter instance for Coordinated Turn (CT) Model
        # Create a KalmanFilter instance for Coordinated Turn (CT) Model
        kf_ct = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        
        def create_ct_model(omega):
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
            kf_ct.P = np.eye(state_dim) * self.params['P']
            kf_ct.Q = np.eye(state_dim) * self.params['q']
            # Specifically, for the turn rate component
            # Assuming omega_variance is an additional hyperparameter
            kf_ct.Q[6, 6] = q * omega_variance
            kf_ct.Q[7, 7] = q * omega_variance
            kf_ct.Q[9, 9] = q * omega_variance  # Higher uncertainty in the turn rate

            kf_ct.R = np.eye(measurement_dim) * self.params['r']
            kf_ct.x = initial_state.copy()
            return kf_ct
        
        kf_cv1 = create_ct_model(initial_state[9])
        kf_cv2 = create_ct_model(0.4)
        kf_cv3 = create_ct_model(0.7)

        # Define initial mode probabilities
        mu = np.array([0.25, 0.25, 0.25, 0.25])

        # Define Markov chain transition matrix
        M = np.array(self.params['M'])  # High likelihood of staying in CT


        # Create IMM Estimator
        return IMMEstimator([kf_cv, kf_cv1, kf_cv2, kf_cv3], mu, M)
    
    def smooth(self, predictions, alpha=0.5):
        smoothed_predictions = np.zeros_like(predictions)
        smoothed_predictions[0] = predictions[0]

        # Forward pass
        for t in range(1, len(predictions)):
            smoothed_predictions[t] = alpha * predictions[t] + (1 - alpha) * smoothed_predictions[t-1]

        return smoothed_predictions

    def predict(self, batch_positions):
        batch_size, timesteps, _ = batch_positions.shape
        
        predictions = np.zeros((batch_size, timesteps, 10))  # Assuming 10 state dimensions
        dt = self.params['dt']
        
        for i in range(batch_size):
            # Initial state estimation
            initial_position = batch_positions[i, 0]
            initial_velocity = (batch_positions[i, 1] - batch_positions[i, 0]) / dt
            initial_acceleration = np.zeros(3)
            initial_turn_rate = 0.1  # Assuming starting with no turn
            initial_state = np.hstack((initial_position, initial_velocity, initial_acceleration, initial_turn_rate))

            # Create IMM estimator with the provided hyperparameters
            imm = self.create_imm_cvct_estimator(initial_state)
            
            # Running the IMM estimator for the first 5 timesteps
            for t in range(5):
                z = batch_positions[i, t]
                imm.predict()
                imm.update(z)
                predictions[i, t] = imm.x

            # Apply a refined smoothing technique to the first 10 timesteps
            predictions[i, :5, :] = self.smooth(predictions[i, :5, :], alpha=0.6)
            
            # Predicting the next 10 timesteps without updating (no measurements)
            for t in range(5, 15):
                imm.predict()
                predictions[i, t] = imm.x

            # Apply smoothing to the last 5 timesteps of predictions
            predictions[i, 5:timesteps, :] = self.smooth(predictions[i, 5:timesteps, :], alpha=0.7)

        return predictions
    
    def hyperparameter_tuning(self, batch_positions):
        # Define the parameter grid
        param_grid = {
            'q': [0.1, 0.2, 0.5, 0.8, 1.0], 
            'r': [0.1, 0.2, 0.5, 0.8, 1.0], 
            'P':[0.1, 0.5, 1.0, 10.0], 
            'M': [[[0.9, 0.1],[0.1, 0.9]],
                [[0.8, 0.2],[0.2, 0.8]],
                [[0.6, 0.4],[0.4, 0.6]],
                [[0.5, 0.5],[0.5, 0.5]]
                                ], 
            'dt':  [0.1,0.4,0.9], 
            'omega_variance':[0.1, 0.2, 0.5, 0.8, 1.0]

        }

        # Generate combinations of parameters
        grid = ParameterGrid(param_grid)

        best_score = float('inf')
        best_params = None

        # Grid search loop
        for params in grid:
            # Initialize the IMMParticleFilter with current parameters
            model = IMM_CVCT_2D(pos_dim=2)
            model.params['q'] = params['q']
            model.params['r'] = params['r']
            model.params['P'] = params['P']
            model.params['M'] = params['M']
            model.params['dt'] = params['dt']
            model.params['omega_variance'] = params['omega_variance']
            
            # Run the filter on your data and calculate the prediction error
            predictions = model.predict(batch_positions[:])
            
            # Calculate the error using the modified function
            error = self.calculate_meanADE(batch_positions[:], predictions, dim=3)

            # Update the best parameters if the current configuration yields a lower error
            if error < best_score:
                best_score = error
                best_params = params

        # Print the best hyperparameters and corresponding score
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        self.params = best_params


class IMM_CVCT_2D(Bayes):
    """ 
    Interacting Multiple Models with Constant Velocity (CV) and Constant Turn (CT) 
    for 2D Position Tracking.
    """
    def __init__(self, pos_dim, name="IMM_CVCT"):
        super(IMM_CVCT_2D, self).__init__(pos_dim=pos_dim, name=name)
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 1, 
                        'M': [[0.70, 0.1, 0.1, 0.1],  # High likelihood of staying in CV
                             [0.1, 0.7, 0.1, 0.1],
                             [0.1, 0.1, 0.7, 0.1],
                             [0.1, 0.1, 0.1, 0.7]], 
                        'dt': 0.4, 
                        'omega_variance': 0.1}

    def create_imm_cvct_estimator(self, initial_state):
        # Define the state and measurement dimensions
        state_dim = 7  # Consistent 7-dimensional state
        measurement_dim = 2  # Assume 2D position measurements (x, y)
        
        dt = self.params['dt']
        omega_variance = self.params['omega_variance']
        
        # Create Kalman Filter for Constant Velocity (CV) Model
        kf_cv = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        kf_cv.F = np.array([[1, 0, dt, 0, 0, 0, 0], # x
                            [0, 1, 0, dt, 0, 0, 0], # y
                            [0, 0, 1, 0, 0, 0, 0], # vx
                            [0, 0, 0, 1, 0, 0, 0], # vy
                            [0, 0, 0, 0, 1, 0, 0], # zero acceleration
                            [0, 0, 0, 0, 0, 1, 0], # zero acceleration
                            [0, 0, 0, 0, 0, 0, 1]]) # constant turn rate
        
        kf_cv.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0]])
        kf_cv.P = np.eye(state_dim) * self.params['P']
        kf_cv.Q = np.eye(state_dim) * self.params['q']
        kf_cv.R = np.eye(measurement_dim) * self.params['r']
        kf_cv.x = np.hstack((initial_state[:4], [0], [0], [0]))  # Expand initial state for CV

        def ct_jacobian( x, dt, omega):
            F = np.eye(7)  # Start with an identity matrix for stability

            v_x = x[2]
            v_y = x[3]

            # If omega is near zero, use a linear approximation
            if abs(omega) < 1e-5:
                F[0, 2] = dt
                F[1, 3] = dt
                F[2, 4] = dt
                F[3, 5] = dt
            else:
                F[0, 2] = np.sin(omega * dt) / omega
                F[0, 3] = -(1 - np.cos(omega * dt)) / omega
                F[1, 2] = (1 - np.cos(omega * dt)) / omega
                F[1, 3] = np.sin(omega * dt) / omega

                F[2, 2] = np.cos(omega * dt)
                F[2, 3] = -np.sin(omega * dt)
                F[3, 2] = np.sin(omega * dt)
                F[3, 3] = np.cos(omega * dt)

            # Acceleration contributions (could be simplified or refined)
            F[2, 4] = dt  # dv_x / da_x
            F[3, 5] = dt  # dv_y / da_y

            # Turn rate affects velocity
            F[2, 6] = -v_y * dt
            F[3, 6] = v_x * dt

            return F
        
        def create_kf_ct(omega):

            # Create Kalman Filter for Coordinated Turn (CT) Model
            kf_ct = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
            kf_ct.F = ct_jacobian(kf_ct.x, dt, omega)
            kf_ct.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0, 0]])
            kf_ct.P = np.eye(state_dim) * self.params['P']
            kf_ct.Q = np.eye(state_dim) * self.params['q']
            kf_ct.Q[4, 4] *= omega_variance  # Adjust variance for the turn rate
            kf_ct.Q[5, 5] *= omega_variance  # Adjust variance for the turn rate
            kf_ct.Q[6, 6] *= omega_variance  # Adjust variance for the turn rate
            kf_ct.R = np.eye(measurement_dim) * self.params['r']
            kf_ct.x = initial_state.copy()  # Use full initial state for CT
            return kf_ct
        
        kf_ct_1 = create_kf_ct(initial_state[6])
        kf_ct_2 = create_kf_ct(0.4)
        kf_ct_3 = create_kf_ct(0.7)

        # Define initial mode probabilities
        mu = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Define Markov chain transition matrix
        M = np.array(self.params['M'])  # Transition probabilities
        
        # Create IMM Estimator
        return IMMEstimator([kf_cv, kf_ct_1, kf_ct_2, kf_ct_3], mu, M)

    def smooth(self, predictions, alpha=0.5):
        smoothed_predictions = np.zeros_like(predictions)
        smoothed_predictions[0] = predictions[0]

        # Forward pass
        for t in range(1, len(predictions)):
            smoothed_predictions[t] = alpha * predictions[t] + (1 - alpha) * smoothed_predictions[t-1]

        return smoothed_predictions

    def predict(self, batch_positions):
        batch_size, timesteps, _ = batch_positions.shape
        prediction_timesteps = 15  # 10 updates + 5 predictions
        predictions = np.zeros((batch_size, prediction_timesteps, 2))  # Predicting only (x, y) positions

        dt = self.params['dt']
        for i in range(batch_size):
            initial_position = batch_positions[i, 0]
            initial_velocity = (batch_positions[i, 1] - batch_positions[i, 0]) / dt
            initial_acceleration = np.zeros(2)
            initial_turn_rate = 0.1  # Assuming starting with no turn

            initial_state = np.hstack((initial_position, initial_velocity, initial_acceleration, initial_turn_rate))
            imm = self.create_imm_cvct_estimator(initial_state)

            # IMM prediction and update for the first 10 timesteps
            for t in range(5):
                z = batch_positions[i, t]
                imm.predict()
                imm.update(z)
                predictions[i, t] = imm.x[:2]

            # Apply a refined smoothing technique to the first 10 timesteps
            predictions[i, :5, :] = self.smooth(predictions[i, :5, :], alpha=0.6)

            # Predicting the next 5 timesteps without measurement updates
            for t in range(5, prediction_timesteps):
                imm.predict()
                predictions[i, t] = imm.x[:2]

            # Apply smoothing to the last 5 timesteps of predictions
            predictions[i, :prediction_timesteps, :] = self.smooth(predictions[i, :prediction_timesteps, :], alpha=0.6)

        return predictions

    def hyperparameter_tuning(self, batch_positions):
        # Define the parameter grid
        param_grid = {
            'q': [0.1, 0.5, 0.8, 1.0], 
            'r': [0.1, 0.5, 0.8, 1.0], 
            'P':[0.1, 0.5, 1.0],
            'dt':  [0.1,0.4,0.9], 
            'omega_variance':[0.1, 0.5, 0.8, 1.0]

        }

        # Generate combinations of parameters
        grid = ParameterGrid(param_grid)

        best_score = float('inf')
        best_params = None

        # Grid search loop
        for params in grid:
            # Initialize the IMMParticleFilter with current parameters
            model = IMM_CVCT_2D(pos_dim=2)
            model.params['q'] = params['q']
            model.params['r'] = params['r']
            model.params['P'] = params['P']
            model.params['dt'] = params['dt']
            model.params['omega_variance'] = params['omega_variance']
            
            # Run the filter on your data and calculate the prediction error
            predictions = model.predict(batch_positions)
            
            # Calculate the error using the modified function
            error = self.calculate_meanADE(batch_positions, predictions, dim=2)

            # Update the best parameters if the current configuration yields a lower error
            if error < best_score:
                best_score = error
                best_params = params

        # Print the best hyperparameters and corresponding score
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        self.params = best_params









