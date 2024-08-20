import os
import sys
import numpy as np 
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

tf.random.set_seed(1234)

def get_relative_dist(l):
    new_l = [l[i] - l[i-1] for i in range(1,len(l))]
    # to keep 15 sequence length add first velocity 0,0,0
    new_l.insert(0,[0.0,0.0,0.0])
    return new_l

def load_data(data_path, batch_size=32):
    # load data pickle
    df = pd.read_pickle(data_path) 
    df.reset_index(inplace=True, drop=True)
    df = df.sample(frac=1).reset_index(drop=True)

    num_steps = 15
    #scale = 100

    # create tf dataset from pandas dataframe
    position_list = []
    pose_list = []
    for index, row in df.iterrows():
        if row['positions'].shape[0] >= num_steps:
            pos_data = row['positions']
            poses_data = row['poses'].reshape(-1,17*3)
            # vox_data = row['voxelgrids'].reshape(-1,1000*2)
            pos_df = tf.data.Dataset.from_tensor_slices(pos_data)
            pose_df = tf.data.Dataset.from_tensor_slices(poses_data)
            pos_df = pos_df.window(num_steps, shift=3, drop_remainder=True)
            pose_df = pose_df.window(num_steps, shift=3, drop_remainder=True)
            for windows1, windows2 in zip(pos_df, pose_df):
                np_arr1 = np.asarray([item.numpy() for item in windows1])
                #arr = get_relative_dist(np_arr1*scale)
                position_list.append(np_arr1)
                np_arr2 = np.asarray([item.numpy() for item in windows2])
                pose_list.append(np_arr2)

    pos_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(position_list))
    pose_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(pose_list))
    zip_ds = tf.data.Dataset.zip((pos_dataset, pose_dataset))
    zip_ds = zip_ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    zip_ds = zip_ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE) 

    # shuffle, batch and split dataset
    DATASET_SIZE = len(zip_ds)

    train_size = int(0.8 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    train_dataset = zip_ds.take(train_size)
    test_dataset = zip_ds.skip(train_size)

    # returns each dataset with tuple of (batch position, batch keypoints)
    return train_dataset, test_dataset

def load_synthetic_data():
    # Generate a sequence of 2D positions with constant acceleration
    def generate_linear_trajectory(length, x_start, y_start, ax, ay):
        x, y = x_start, y_start
        z = np.random.uniform(0.5, 2)
        vx, vy = 0, 0
        trajectory = [(x, y,z)]
        for _ in range(length - 1):
            vx += ax
            vy += ay
            x += vx
            y += vy
            trajectory.append((x, y,z))
        return trajectory

    # Generate a sequence of 2D positions representing a circular trajectory
    def generate_circular_trajectory(length, radius, cx, cy, angular_velocity):
        trajectory = []
        z = np.random.uniform(0.5, 2)
        for t in range(length):
            angle = angular_velocity * t
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            trajectory.append((x, y,z))
        return trajectory

    # Prepare data for the LSTM
    def get_dataset(seq_len, n_samples):
        traj = list()
        for _ in range(n_samples):
            # Randomly choose between linear or circular trajectory
            if np.random.rand() < 0.5:
                # Generate linear trajectory
                x_start, y_start = np.random.uniform(0, 100), np.random.uniform(0, 100)
                ax, ay = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
                source = generate_linear_trajectory(seq_len, x_start, y_start, ax, ay)
            else:
                # Generate circular trajectory
                radius = np.random.uniform(5, 20)
                cx, cy = np.random.uniform(20, 60), np.random.uniform(20, 60)
                angular_velocity = np.random.uniform(0.1, 0.3)
                source = generate_circular_trajectory(seq_len, radius, cx, cy, angular_velocity)

            traj.append(source)
        return traj

    ds_pos = get_dataset(15, 320000)
    ds_pose = np.ones((320000,15,51))

    pos_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(ds_pos))
    pose_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(ds_pose))
    zip_ds = tf.data.Dataset.zip((pos_dataset, pose_dataset))
    zip_ds = zip_ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    zip_ds = zip_ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE) 

    # shuffle, batch and split dataset
    DATASET_SIZE = len(zip_ds)

    train_size = int(0.8 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    train_dataset = zip_ds.take(train_size)
    test_dataset = zip_ds.skip(train_size)

    return train_dataset, test_dataset


def load_synthetic_data2d():
    # Generate a sequence of 2D positions with constant acceleration
    def generate_linear_trajectory(length, x_start, y_start, ax, ay):
        x, y = x_start, y_start
        vx, vy = 0, 0
        trajectory = [(x, y)]
        for _ in range(length - 1):
            vx += ax
            vy += ay
            x += vx
            y += vy
            trajectory.append((x, y))
        return trajectory

    # Generate a sequence of 2D positions representing a circular trajectory
    def generate_circular_trajectory(length, radius, cx, cy, angular_velocity):
        trajectory = []
        for t in range(length):
            angle = angular_velocity * t
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            trajectory.append((x, y))
        return trajectory

    # Prepare data for the LSTM
    def get_dataset(seq_len, n_samples):
        traj = list()
        for _ in range(n_samples):
            # Randomly choose between linear or circular trajectory
            if np.random.rand() < 0.5:
                # Generate linear trajectory
                x_start, y_start = np.random.uniform(0, 100), np.random.uniform(0, 100)
                ax, ay = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
                source = generate_linear_trajectory(seq_len, x_start, y_start, ax, ay)
            else:
                # Generate circular trajectory
                radius = np.random.uniform(5, 20)
                cx, cy = np.random.uniform(20, 60), np.random.uniform(20, 60)
                angular_velocity = np.random.uniform(0.1, 0.3)
                source = generate_circular_trajectory(seq_len, radius, cx, cy, angular_velocity)

            traj.append(source)
        return traj

    ds_pos = get_dataset(15, 320000)
    ds_pose = np.ones((320000,15,51))

    pos_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(ds_pos))
    pose_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(ds_pose))
    zip_ds = tf.data.Dataset.zip((pos_dataset, pose_dataset))
    zip_ds = zip_ds.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    zip_ds = zip_ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE) 

    # shuffle, batch and split dataset
    DATASET_SIZE = len(zip_ds)

    train_size = int(0.8 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    train_dataset = zip_ds.take(train_size)
    test_dataset = zip_ds.skip(train_size)

    return train_dataset, test_dataset


