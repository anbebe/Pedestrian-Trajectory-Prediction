import os
import sys
import numpy as np 
import open3d as o3d
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

tf.random.set_seed(1234)

def load_data(data_path, batch_size=32):
    # load data pickle
    df = pd.read_pickle(data_path) 
    df.reset_index(inplace=True, drop=True)

    num_steps = 15

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
                position_list.append(np_arr1)
                np_arr2 = np.asarray([item.numpy() for item in windows2])
                pose_list.append(np_arr2)

    # 2783 sequences
    pos_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(position_list))
    pose_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(pose_list))
    zip_ds = tf.data.Dataset.zip((pos_dataset, pose_dataset))
    zip_ds = zip_ds.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    zip_ds = zip_ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE) 

    # shuffle, batch and split dataset
    DATASET_SIZE = len(zip_ds)

    train_size = int(0.8 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    train_dataset = zip_ds.take(train_size)
    test_dataset = zip_ds.skip(train_size)

    # returns each dataset with tuple of (batch position, batch keypoints)
    return train_dataset, test_dataset


