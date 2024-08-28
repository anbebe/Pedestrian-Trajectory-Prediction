import numpy as np
from scipy.stats import norm
from numpy.random import uniform, normal
from bayes import Bayes
from sklearn.model_selection import ParameterGrid

class ParticleFilter(Bayes):
    def __init__(self, name="Particle_Filter"):
        super(ParticleFilter, self).__init__(pos_dim=2, name=name)
        self.state_dim = 4
        self.measurement_dim = 2
        
        if self.params == None:
            self.params = {'num_particles': 2000,
                        'process_noise_cv': 0.4, 
                        'measurement_noise': 0.2, 
                        'dt': 0.41}

        # Particles and weights for each model
        self.particles_cv = np.zeros((self.params['num_particles'], self.state_dim))
        self.weights_cv = np.ones(self.params['num_particles']) / self.params['num_particles']


    def create_uniform_particles(self, x_range, y_range, vx_range, vy_range):
        for model_particles in [self.particles_cv, self.particles_ctrv]:
            model_particles[:, 0] = uniform(x_range[0], x_range[1], size=self.params['num_particles'])
            model_particles[:, 1] = uniform(y_range[0], y_range[1], size=self.params['num_particles'])
            model_particles[:, 2] = uniform(vx_range[0], vx_range[1], size=self.params['num_particles'])
            model_particles[:, 3] = uniform(vy_range[0], vy_range[1], size=self.params['num_particles'])

    def estimate_initial_velocity(self, first_two_observations, dt=1.0):
        """
        Estimate initial velocity based on the first two observations.
        """
        velocity = (first_two_observations[1] - first_two_observations[0]) / dt
        return velocity

    def create_particles_with_estimated_velocity(self, first_observation, estimated_velocity, omega_init=0, position_noise=0.1, velocity_noise=0.1, omega_noise=0.01):
        """
        Initialize particles based on the first observation and estimated velocity with added noise.
        """
        for model_particles in [self.particles_cv]:#[self.particles_cv, self.particles_ctrv]:
            # Position initialization around the first observed position
            model_particles[:, 0] = normal(first_observation[0], position_noise, size=self.params['num_particles'])
            model_particles[:, 1] = normal(first_observation[1], position_noise, size=self.params['num_particles'])
            
            # Velocity initialization around the estimated velocity
            model_particles[:, 2] = normal(estimated_velocity[0], velocity_noise, size=self.params['num_particles'])
            model_particles[:, 3] = normal(estimated_velocity[1], velocity_noise, size=self.params['num_particles'])

    def predict_cv(self):
        self.particles_cv[:, 0] += self.particles_cv[:, 2] * self.params['dt'] + normal(0, self.params['process_noise_cv'], size=self.params['num_particles'])
        self.particles_cv[:, 1] += self.particles_cv[:, 3] * self.params['dt'] + normal(0, self.params['process_noise_cv'], size=self.params['num_particles'])

   
    def update(self, z):
        # Compute weights based on measurement likelihood
        diff_cv = self.particles_cv[:, :self.measurement_dim] - z
        dist_cv = np.linalg.norm(diff_cv, axis=1)
        self.weights_cv *= norm.pdf(dist_cv, scale=self.params['measurement_noise'])

        # Normalize weights
        self.weights_cv /= np.sum(self.weights_cv)

    def resample(self):
        indices_cv = np.random.choice(self.params['num_particles'], size=self.params['num_particles'], p=self.weights_cv)
        self.particles_cv = self.particles_cv[indices_cv]
        self.weights_cv.fill(1.0 / self.params['num_particles'])

    def estimate_average(self):
        # Combine the estimates from both models and take average of all particles
        estimate_cv = np.average(self.particles_cv, weights=self.weights_cv, axis=0)
        return estimate_cv
    
    def estimate_top_three(self, top_n=3):
        # Sort particles by weight in descending order
        sorted_indices_cv = np.argsort(self.weights_cv)[::-1]

        # Get the top N particles for each model
        top_particles_cv = self.particles_cv[sorted_indices_cv[:top_n]]
    
        # Combine the top N particles based on model probabilities
        combined_estimates = []
        for i in range(top_n):
            combined_estimates.append(top_particles_cv[i])
        
        return np.array(combined_estimates)

    def predict(self, batch_positions):
        batch_size, timesteps, _ = batch_positions.shape
        predictions = np.zeros((batch_size, timesteps, 4))
        
        for i in range(batch_size):
            imm_pf = ParticleFilter()
            first_two_observations = batch_positions[i][:2]
            initial_velocity = imm_pf.estimate_initial_velocity(first_two_observations)
            imm_pf.create_particles_with_estimated_velocity(first_two_observations[0], initial_velocity)

            for t in range(5):
                z = batch_positions[i, t]
                imm_pf.predict_cv()
                imm_pf.update(z)
                imm_pf.resample()
                predictions[i, t] = imm_pf.estimate_average()
            
            for t in range(5, timesteps):
                imm_pf.predict_cv()
                predictions[i, t] = imm_pf.estimate_average()
        
        return predictions
    
    def hyperparameter_tuning(self, batch_positions,index = 0):
        # Define the parameter grid
        param_grid = {
            'num_particles': [500, 1000, 2000, 5000],
            'process_noise_cv': [0.1, 0.2, 0.5, 0.8, 1.0],
            'measurement_noise': [0.2, 0.5, 1.0],
            'dt': [0.1,0.4,0.6]

        }

        # Generate combinations of parameters
        grid = ParameterGrid(param_grid)

        best_score = float('inf')
        best_params = None

        # Grid search loop
        for params in grid:
            # Initialize the IMMParticleFilter with current parameters
            model = ParticleFilter(pos_dim=2)
            model.params['num_particles'] = params['num_particles']
            model.params['process_noise_cv'] = params['process_noise_cv']
            model.params['measurement_noise'] = params['measurement_noise']
            model.params['dt'] = params['dt']
            
            # Run the filter on your data and calculate the prediction error
            predictions = model.predict(batch_positions[index][:])
            
            # Calculate the error using the modified function
            error = self.calculate_meanADE(batch_positions[index][:], predictions, dim=2)

            # Update the best parameters if the current configuration yields a lower error
            if error < best_score:
                best_score = error
                best_params = params

        # Print the best hyperparameters and corresponding score
        print("Best Parameters:", best_params)
        print("Best Score:", best_score)
        self.params = best_params

