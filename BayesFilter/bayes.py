# Base class
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from itertools import product

class Bayes():
    
    def __init__(self, pos_dim, name=None) -> None:
        self.name = name
        self.pos_dim = pos_dim # if 2d or 3d positions
        self.params = None

    def predict(self, batch_positions):
        pass

    def hyperparameter_tuning(self, batch_positions):
        pass

    def plot_predictions(self, ground_truth, predictions, sample_index):
        gt_x = ground_truth[sample_index, :, 0]
        gt_y = ground_truth[sample_index, :, 1]
        pred_x = predictions[sample_index, :, 0]
        pred_y = predictions[sample_index, :, 1]

        plt.figure(figsize=(10, 6))
        plt.plot(gt_x, gt_y, label='Ground Truth', marker='o')
        plt.plot(pred_x, pred_y, label='Prediction', marker='x')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(self.name)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def calculate_minADE(self, ground_truth, predictions):
        # Extract the position part from predictions (assuming position is the first dim components)
        predicted_positions = predictions[:, :, :self.pos_dim]
        # Calculate the l2 distance between each point in the trajectory
        displacement_errors = np.linalg.norm(ground_truth - predicted_positions, axis=2)
        # Calculate the average displacement error
        minADE = np.min(np.mean(displacement_errors, axis=1))
        return minADE
    
    def calculate_meanADE(self, ground_truth, predictions, dim):
        predicted_positions = predictions[:, :, :dim]
        displacement_errors = np.linalg.norm(ground_truth - predicted_positions, axis=2)
        ADE = np.mean(np.mean(displacement_errors, axis=1))
        return ADE

    def calculate_minFDE(self, ground_truth, predictions, dim):
        # Extract the position part from predictions (assuming position is the first dim components)
        predicted_positions = predictions[:, :, :dim]
        # Calculate the l2 distance between the final positions
        final_displacement_errors = np.linalg.norm(ground_truth[:, -1] - predicted_positions[:, -1], axis=1)
        # Find the minimum final displacement error
        minFDE = np.min(final_displacement_errors)
        return minFDE
    
    def calculate_meanFDE(self, ground_truth, predictions):
        predicted_positions = predictions[:, :, :self.pos_dim]
        final_displacement_errors = np.linalg.norm(ground_truth[:, -1] - predicted_positions[:, -1], axis=1)
        FDE = np.mean(final_displacement_errors)
        return FDE
    
    def hyperparameter_tuning_(self, batch_positions):
        # Define the hyperparameter space
        q_values = [0.01, 0.1, 0.5]  
        r_values = [0.001, 0.01, 0.1] 
        P_values = [1, 10, 100] 
        M_values = [
            np.array([[0.85, 0.15], [0.15, 0.85]]),  
            np.array([[0.7, 0.3], [0.3, 0.7]]),
            np.array([[0.6, 0.4], [0.4, 0.6]]) 
        ]
        dt_values = [0.2, 0.5, 1.0]  
        omega_variance_values = [0.01, 0.1, 1.0]  

        # Track the best hyperparameters and metrics
        best_hyperparameters = None
        best_minADE = float('inf')
        best_minFDE = float('inf')

        # Loop through all combinations of hyperparameters
        for omega_variance in omega_variance_values:
            for q, r, P, M, dt in product(q_values, r_values, P_values, M_values, dt_values):
            
                # Run the IMM estimator with the current hyperparameters
                predictions = self.predict(batch_positions[0][0:], q, r, omega_variance, P, M, dt)
                
                # Calculate metrics
                minADE = self.calculate_minADE(batch_positions[0][0:], predictions)
                minFDE = self.calculate_minFDE(batch_positions[0][0:], predictions)
                
                # Update the best hyperparameters if this combination is better
                if minADE < best_minADE or (minADE == best_minADE and minFDE < best_minFDE):
                    best_minADE = minADE
                    best_minFDE = minFDE
                    best_hyperparameters = {'q': q, 'r': r, 'P': P, 'M': M, 'dt': dt, 'omega_variance': omega_variance}

        # Output the best hyperparameters and corresponding metrics
        print("Best Hyperparameters:", best_hyperparameters)
        print("Best minADE:", best_minADE)
        print("Best minFDE:", best_minFDE)
        self.hyperparameters = best_hyperparameters


def load_dataset(input_path, batch_size, pos_dim, scale=1):
        loaded = tf.data.experimental.load(input_path)
        def tf_dataset_to_numpy(tf_dataset):
            numpy_data = []
            for batch in tf_dataset.as_numpy_iterator():
                if batch[0].shape[0]==batch_size:
                    numpy_data.append(batch[0][...,:pos_dim])
            return np.asarray(numpy_data)*scale

        # Convert the TensorFlow dataset to a numpy array
        return tf_dataset_to_numpy(loaded)



