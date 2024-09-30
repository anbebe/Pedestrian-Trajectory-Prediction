import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.kalman import IMMEstimator
from bayes import Bayes
from sklearn.model_selection import ParameterGrid



class IMM_CVCT_2D(Bayes):
    """ 
    Interacting Multiple Models with Constant Velocity (CV) and Constant Turn (CT) 
    for predicting trajectories in 2D.
    """
    def __init__(self, pos_dim, name="IMM_CVCT"):
        super(IMM_CVCT_2D, self).__init__(pos_dim=pos_dim, name=name)
        if self.params == None:
            self.params = {'q': 0.1, 'r': 0.1, 'P': 1, 
                        'M': [[0.70, 0.1, 0.1, 0.1],
                             [0.1, 0.7, 0.1, 0.1],
                             [0.1, 0.1, 0.7, 0.1],
                             [0.1, 0.1, 0.1, 0.7]], 
                        'dt': 0.4, 
                        'omega_variance': 0.1}

    def create_imm_cvct_estimator(self, initial_state):
        """
        Creates the CV and CT models to be used in the IMM and initialise the state
        from given param.

        :param initial_state initial values of position, velocity and turn rate

        :returns IMMEstimator as IMM model
        """
       
        state_dim = 5 
        # assume 2D position measurements (x, y)
        measurement_dim = 2  
        
        dt = self.params['dt']
        omega_variance = self.params['omega_variance']
        
        # create KF for constant velocity (CV) model
        kf_cv = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
        # state transition matrix
        kf_cv.F = np.array([[1, 0, dt, 0, 0], 
                            [0, 1, 0, dt, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]])
        # measurement matrix
        kf_cv.H = np.array([[1, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0]])
        # initial state covariance
        kf_cv.P = np.eye(state_dim) * self.params['P']
        # process noise covariance
        kf_cv.Q = np.eye(state_dim) * self.params['q']
        #m easurement noise covariance
        kf_cv.R = np.eye(measurement_dim) * self.params['r']
        # initial state
        kf_cv.x = initial_state

        def ct_jacobian(dt, omega):
            """
                create state transition matrix for constant turn by creating the
                linear jacobian from the rotation 
            """

            sin_omega_dt = np.sin(omega * dt)
            cos_omega_dt = np.cos(omega * dt)
            
            F = np.array([
                [1, 0, sin_omega_dt/omega, -(1-cos_omega_dt)/omega, 0],
                [0, 1, (1-cos_omega_dt)/omega, sin_omega_dt/omega, 0],
                [0, 0, cos_omega_dt, -sin_omega_dt, 0],
                [0, 0, sin_omega_dt, cos_omega_dt, 0],
                [0, 0, 0, 0, 1]
            ])
            return F
        
        def create_kf_ct(omega):
            """
            create KF for constant (CT) model
            """
            kf_ct = KalmanFilter(dim_x=state_dim, dim_z=measurement_dim)
            # state transition matrix
            kf_ct.F = ct_jacobian(dt, omega)
            # measurement matrix
            kf_ct.H = np.array([[1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0]])
            # initial state covariance
            kf_ct.P = np.eye(state_dim) * self.params['P']
            # process noise covariance
            kf_ct.Q = np.eye(state_dim) * self.params['q']
            # Adjust variance for the turn rate
            kf_ct.Q[4, 4] *= omega_variance  
            #m easurement noise covariance
            kf_ct.R = np.eye(measurement_dim) * self.params['r']
            # initial state
            kf_ct.x = initial_state
            return kf_ct
        
        # create ct models wih different fixed turn rates
        kf_ct_1 = create_kf_ct(initial_state[4])
        kf_ct_2 = create_kf_ct(0.4)
        kf_ct_3 = create_kf_ct(0.7)

        # Define initial mode probabilities
        mu = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Define Markov chain transition matrix
        M = np.array(self.params['M']) 
        
        # Create IMM Estimator
        return IMMEstimator([kf_cv, kf_ct_1, kf_ct_2, kf_ct_3], mu, M)

    def smooth(self, predictions, alpha=0.5):
        """
        Smooth predictions that are previously made only on current observations by taking
        the predictions made before into account

        :param predictions sequence of predicted coordinates
        :param alpha how much the predictions are adpated to the previous steps

        :return smoothed predictions
        """
        smoothed_predictions = np.zeros_like(predictions)
        smoothed_predictions[0] = predictions[0]

        for t in range(1, len(predictions)):
            smoothed_predictions[t] = alpha * predictions[t] + (1 - alpha) * smoothed_predictions[t-1]

        return smoothed_predictions

    def predict(self, batch_positions):
        """
        Predict the next 10 timesteps from 5 given the IMM model

        :param batch_positions ground truth trajectories of shape (batch_size, 15, 2)

        :returns predictions of shape  (batch_size, 15, 2)
        """

        # history steps
        tmp_t =5
        batch_size, timesteps, _ = batch_positions.shape
        prediction_timesteps = 15 
        predictions = np.zeros((batch_size, prediction_timesteps, 2)) 
        dt = self.params['dt']

        for i in range(batch_size):
            initial_position = batch_positions[i, 0]
            initial_velocity = (batch_positions[i, 1] - batch_positions[i, 0]) / dt
            initial_turn_rate = 0.1

            initial_state = np.hstack((initial_position, initial_velocity, initial_turn_rate))
            imm = self.create_imm_cvct_estimator(initial_state)

            # predict and update for the first 5 timesteps
            for t in range(tmp_t):
                z = batch_positions[i, t]
                imm.predict()
                imm.update(z)
                predictions[i, t] = imm.x[:2]

            # smooth first 5 predictions
            predictions[i, :tmp_t, :] = self.smooth(predictions[i, :tmp_t, :], alpha=0.6)

            # predict the next 10 timesteps without measurement updates
            for t in range(tmp_t, prediction_timesteps):
                imm.predict()
                predictions[i, t] = imm.x[:2]

            # smooth predictions
            predictions[i, :prediction_timesteps, :] = self.smooth(predictions[i, :prediction_timesteps, :], alpha=0.6)

        return predictions

    def hyperparameter_tuning(self, batch_positions):
        """
            Applies hyperparameter tuning by trying the KF with different sets of parameters 
            and setting the internal parameters to the set achieving the best ADE.

            :param batch_positions tarjectories with the shape (batch_size, sequence_length, 2)
        """
        # Define the parameter grid
        param_grid = {
            'q': [0.1, 0.5, 0.8, 1.0], 
            'r': [0.1, 0.5, 0.8, 1.0], 
            'P':[0.1, 0.5, 1.0],
            'dt':  [0.1,0.4,0.9], 
            'omega_variance':[0.1, 0.5, 0.8, 1.0]

        }

        grid = ParameterGrid(param_grid)
        best_score = float('inf')
        best_params = None

        # Loop through all combinations of hyperparameters
        for params in grid:
            model = IMM_CVCT_2D(pos_dim=2)
            model.params['q'] = params['q']
            model.params['r'] = params['r']
            model.params['P'] = params['P']
            model.params['dt'] = params['dt']
            model.params['omega_variance'] = params['omega_variance']
            
            predictions = model.predict(batch_positions)
            
            error = self.calculate_meanADE(batch_positions, predictions)

            # Update the best parameters
            if error < best_score:
                best_score = error
                best_params = params

        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        self.params = best_params









