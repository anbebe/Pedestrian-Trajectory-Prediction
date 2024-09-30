import numpy as np
from scipy.stats import norm
from numpy.random import uniform, normal
from bayes import Bayes
from sklearn.model_selection import ParameterGrid

class ParticleFilter(Bayes):

    """
        Particle Filter Class that uses the constant velocity model to predict trajectories.
    """

    def __init__(self, name="Particle_Filter"):
        super(ParticleFilter, self).__init__(pos_dim=2, name=name)
        self.state_dim = 4
        self.measurement_dim = 2
        
        if self.params == None:
            self.params = {'num_particles': 2000,
                        'process_noise_cv': 0.4, 
                        'measurement_noise': 0.2, 
                        'dt': 0.41}

        # Particles and weights for the cv model
        self.particles = np.zeros((self.params['num_particles'], self.state_dim))
        self.weights = np.ones(self.params['num_particles']) / self.params['num_particles']


    def estimate_initial_velocity(self, observations, dt=1.0):
        """
        Estimate initial velocity based on the first two observations.

        :param observations the first two positions of the trajectory of shape (2,2)

        :returns estimated initial velocity vector of shape (2)
        """
        velocity = (observations[1] - observations[0]) / self.params['dt']
        return velocity

    def create_particles_with_estimated_velocity(self, init_pos, init_velo, position_noise=0.1, velocity_noise=0.1):
        """
        Initialize particles based on the first observation and estimated velocity with added noise.

        :param init_pos the first trajectory position in 2D
        :param init_velo the first velocity of the trajectory in 2D
        :param position_noise initial position noise
        :param init_velo initial velocity noise
        """
        for model_particles in [self.particles]:
            # add noise to the postion initialization
            model_particles[:, 0] = normal(init_pos[0], position_noise, size=self.params['num_particles'])
            model_particles[:, 1] = normal(init_pos[1], position_noise, size=self.params['num_particles'])
            
            # add noise to the velocity initialization
            model_particles[:, 2] = normal(init_velo[0], velocity_noise, size=self.params['num_particles'])
            model_particles[:, 3] = normal(init_velo[1], velocity_noise, size=self.params['num_particles'])

    def predict_particles(self):
        """
        Predict the next state of each particle with the CV model
        """
        self.particles[:, 0] += self.particles[:, 2] * self.params['dt'] + normal(0, self.params['process_noise_cv'], size=self.params['num_particles'])
        self.particles[:, 1] += self.particles[:, 3] * self.params['dt'] + normal(0, self.params['process_noise_cv'], size=self.params['num_particles'])

   
    def update(self, z):
        """
        Update the particle weights based on the measurement likelihood

        :param z current measurement of position
        """
        # Compute distance bewtween particles belief and measurement
        diff_cv = self.particles[:, :self.measurement_dim] - z
        dist_cv = np.linalg.norm(diff_cv, axis=1)
        # Adjusts weights by the gaussian likelihood
        self.weights *= norm.pdf(dist_cv, scale=self.params['measurement_noise'])

        # Normalize weights
        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        Resample the particles based on their wights using systematic resampling
        """
        # handle nan and zero values
        self.weights = np.nan_to_num(self.weights, nan=0.0)
        if np.sum(self.weights) == 0:
            self.weights.fill(1.0 / self.params['num_particles'])
        else:
            self.weights /= np.sum(self.weights)

        # resample weights based on weights and reset the weights
        indices_cv = np.random.choice(self.params['num_particles'], size=self.params['num_particles'], p=self.weights)
        self.particles = self.particles[indices_cv]
        self.weights.fill(1.0 / self.params['num_particles'])

    def estimate_average(self):
        """
        Estimates the average of the predicted states of all particles.
        """
        estimate_cv = np.average(self.particles, weights=self.weights, axis=0)
        return estimate_cv


    def predict(self, batch_positions):
        """
        Predict the next 10 timesteps from 5 given ground truth 2D positions with the Particle filter and
        a constant velocity model.

        :param batch_positions ground truth trajectories of shape (batch_Size, 15,2)

        :returns predicted trajectory of shape (batch_size, 15,2)
        """
        # history steps
        tmp_t = 5
        batch_size, timesteps, _ = batch_positions.shape
        predictions = np.zeros((batch_size, timesteps, 4))
        
        for i in range(batch_size):
            imm_pf = ParticleFilter()
            first_two_observations = batch_positions[i][:2]
            initial_velocity = imm_pf.estimate_initial_velocity(first_two_observations)
            imm_pf.create_particles_with_estimated_velocity(first_two_observations[0], initial_velocity)

            for t in range(tmp_t):
                z = batch_positions[i, t]
                imm_pf.predict_particles()
                imm_pf.update(z)
                imm_pf.resample()
                predictions[i, t] = imm_pf.estimate_average()
            
            for t in range(tmp_t, timesteps):
                imm_pf.predict_particles()
                predictions[i, t] = imm_pf.estimate_average()
        
        return predictions
    
    def hyperparameter_tuning(self, batch_positions):
        """
            Applies hyperparameter tuning by trying the PF with different sets of parameters 
            and setting the internal parameters to the set achieving the best ADE.

            :param batch_positions tarjectories with the shape (batch_size, sequence_length, 2)
        """
        # Define the parameter grid
        param_grid = {
            'num_particles': [1000, 2000, 5000],
            'process_noise_cv': [0.2, 0.5, 1.0],
            'measurement_noise': [0.2, 0.5, 1.0],
            'dt': [0.1,0.5,1.0]

        }

        grid = ParameterGrid(param_grid)
        best_score = float('inf')
        best_params = None

        for params in grid:
            self.params['num_particles'] = params['num_particles']
            self.params['process_noise_cv'] = params['process_noise_cv']
            self.params['measurement_noise'] = params['measurement_noise']
            self.params['dt'] = params['dt']
            
            predictions = self.predict(batch_positions[:])
            
            error = self.calculate_meanADE(batch_positions[:], predictions)

            # Update the best parameters
            if error < best_score:
                best_score = error
                best_params = params

        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        self.params = best_params

