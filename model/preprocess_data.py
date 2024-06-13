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

    # create tf dataset from pandas dataframe
    dataset_list = []
    for index, row in df.iterrows():
        if row['positions'].shape[0] >= 20:
            pos_data = row['positions']
            poses_data = row['poses'].reshape(-1,17*3)
            vox_data = row['voxelgrids'].reshape(-1,1000*2)
            data_df = tf.data.Dataset.from_tensor_slices({"input_pos": pos_data, "input_poses":poses_data,"input_grid": vox_data})
            # split and pad in sequences of 20 timesteps - similar to jrdb 9 historic and 19 in total
            data_df = data_df.window(20, shift=3, drop_remainder=True)
            for windows in data_df:
                dataset_list.append(tf.data.Dataset.from_tensors(windows))

    dataset = tf.data.Dataset.from_tensor_slices(dataset_list) # 1201 sequences

    # create mask array, True= needs to be predicted
    mask_arrays = []
    for i in range(len(dataset)):
        mask_arr = [False] * 10 + [True] *10
        # hide 0-3 in between steps (for lazyness whole datapoint)
        hidden_nr = np.random.randint(4)
        hidden_idx = np.random.choice(range(10),hidden_nr, replace=False)
        for i in hidden_idx:
            mask_arr[i] = True
        mask_arrays.append(mask_arr)

    # shuffle, batch and split dataset
    SEED = 42
    batch_size = 32

    full_dataset = dataset.shuffle(100,SEED).batch(batch_size)

    DATASET_SIZE = len(full_dataset)

    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.15 * DATASET_SIZE)
    test_size = int(0.15 * DATASET_SIZE)

    train_dataset = full_dataset.take(train_size)
    test_dataset = full_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)



