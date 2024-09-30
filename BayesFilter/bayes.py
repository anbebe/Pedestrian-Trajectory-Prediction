# Base class
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from itertools import product

class Bayes():

    """
        Parent class for different Bayes filter that predict trajectories in 2D or 3D
    """
    
    def __init__(self, pos_dim, name=None) -> None:
        self.name = name
        self.pos_dim = pos_dim # if 2d or 3d positions
        self.params = None

    def predict(self, batch_positions):
        pass

    def hyperparameter_tuning(self, batch_positions):
        pass

    def plot_predictions(self, ground_truth, predictions, sample_index):
        """
            Plot two trajectories (ground truth and predictions) in one plot taking the first
            two components.

            :param ground_truth: ground truth trajectories of positions (shape [n, sequence_length, coordinates])
            :param predictions: prediction trajectories of positions (shape [n, sequence_length, coordinates])
            :param sample_index: index of trajectory to plot in range 0 - n
        """
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
        """
        Calculates the minimal ADE (average displacement error) over multiple trajectories from
        corresponding ground_truth and predictions. Assumes same dim of ground truth and predictions.

        :param ground_truth: ground truth trajectories of positions (shape [n, sequence_length, coordinates])
        :param predictions: prediction trajectories of positions (shape [n, sequence_length, coordinates])

        :return minimal ADE value
        """
        predicted_positions = predictions[:, :, :self.pos_dim]
        # Calculate the l2 distance between each point in the trajectory
        displacement_errors = np.linalg.norm(ground_truth - predicted_positions, axis=2)
        # Calculate the minimum over all ADEs
        minADE = np.min(np.mean(displacement_errors, axis=1))
        return minADE
    
    def calculate_meanADE(self, ground_truth, predictions):
        """
        Calculates the average ADE (average displacement error) over multiple trajectories from
        corresponding ground_truth and predictions. Assumes same dim of ground truth and predictions.

        :param ground_truth: ground truth trajectories of positions (shape [n, sequence_length, coordinates])
        :param predictions: prediction trajectories of positions (shape [n, sequence_length, coordinates])

        :return average ADE value
        """
        predicted_positions = predictions[:, :, :self.pos_dim]
        # Calculate the l2 distance between each point in the trajectory
        displacement_errors = np.linalg.norm(ground_truth - predicted_positions, axis=2)
        # Calculate the average of all ADEs
        ADE = np.mean(np.mean(displacement_errors, axis=1))
        return ADE

    def calculate_minFDE(self, ground_truth, predictions):
        """
        Calculates the minimal FDE (final displacement error) over multiple trajectories from
        corresponding ground_truth and predictions. Assumes same dim of ground truth and predictions.

        :param ground_truth: ground truth trajectories of positions (shape [n, sequence_length, coordinates])
        :param predictions: prediction trajectories of positions (shape [n, sequence_length, coordinates])

        :return minimal ADE value
        """
        predicted_positions = predictions[:, :, :self.pos_dim]
        # Calculate the l2 distance between the final positions
        final_displacement_errors = np.linalg.norm(ground_truth[:, -1] - predicted_positions[:, -1], axis=1)
        # Find the minimum final displacement error
        minFDE = np.min(final_displacement_errors)
        return minFDE
    
    def calculate_meanFDE(self, ground_truth, predictions):
        """
        Calculates the average FDE (final displacement error) over multiple trajectories from
        corresponding ground_truth and predictions. Assumes same dim of ground truth and predictions.

        :param ground_truth: ground truth trajectories of positions (shape [n, sequence_length, coordinates])
        :param predictions: prediction trajectories of positions (shape [n, sequence_length, coordinates])

        :return minimal ADE value
        """
        predicted_positions = predictions[:, :, :self.pos_dim]
        # Calculate the l2 distance between the final positions
        final_displacement_errors = np.linalg.norm(ground_truth[:, -1] - predicted_positions[:, -1], axis=1)
        # Find the mean final displacement error
        FDE = np.mean(final_displacement_errors)
        return FDE
    
    


def load_dataset(input_path, batch_size, pos_dim, scale=1):
        """
        Loads a tf dataset from a path and converts it to a numpy dataset.

        :param input_path: input path to a tensorflow dataset as string
        :param batch_size: batch_size of the dataset
        :param pos_dim: dimension of the coordinates (2 or 3)
        :param scale: scale parameter for the postions

        :return dataset as numpy array
        """
        loaded = tf.data.experimental.load(input_path)
        def tf_dataset_to_numpy(tf_dataset):
            numpy_data = []
            for batch in tf_dataset.as_numpy_iterator():
                # only get full batches to avoid non mathcing shapes
                if batch[0].shape[0]==batch_size:
                    numpy_data.append(batch[0][...,:pos_dim])
            return np.asarray(numpy_data)*scale

        # Convert the TensorFlow dataset to a numpy array
        return tf_dataset_to_numpy(loaded)



