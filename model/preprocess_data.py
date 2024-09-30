import numpy as np 
import pandas as pd
import tensorflow as tf

tf.random.set_seed(1234)

def get_relative_dist(l):
    """
    Computes the relative distance between single steps of a trajectory in a list assuimng 3D positions

    :param l: list of absolute positions

    :returns list of relative distances starting from (0,0,0) 
    """
    new_l = [l[i] - l[i-1] for i in range(1,len(l))]
    # to keep 15 sequence length add first velocity 0,0,0
    new_l.insert(0,[0.0,0.0,0.0])
    return new_l

def load_data(data_path, batch_size=32):
    """
    Loads data from a saved pandasd dataframe as tuple of positions and poses (skeletal keypoints), 
    and processes it to a tensorflwo dataset with batches

    :param data_path: path to the dataset pickle file

    :returns tensorflow dataset containing tuples of position and pose trajectories
    """
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
        # get only trajectories that have at least 18 timesteps
        if row['positions'].shape[0] >= num_steps:
            pos_data = row['positions']
            poses_data = row['poses'].reshape(-1,17*3)
            # vox_data = row['voxelgrids'].reshape(-1,1000*2)

            # create tensorflow datasets for the features and window them to get more trajectories
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

    #create tensorflow dataset frpm features, zip, batch and shuffle them
    pos_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(position_list))
    pose_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(pose_list))
    zip_ds = tf.data.Dataset.zip((pos_dataset, pose_dataset))
    zip_ds = zip_ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    zip_ds = zip_ds.batch(32).prefetch(tf.data.experimental.AUTOTUNE) 

    # split dataset in train and test
    DATASET_SIZE = len(zip_ds)
    train_size = int(0.8 * DATASET_SIZE)
    test_size = int(0.2 * DATASET_SIZE)

    train_dataset = zip_ds.take(train_size)
    test_dataset = zip_ds.skip(train_size)

    # returns each dataset with tuple of (batch position, batch keypoints)
    return train_dataset, test_dataset

