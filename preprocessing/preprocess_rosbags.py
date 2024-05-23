#global
import argparse
import os
import pandas as pd

# local
from preprocessing.synchronize_features import rosbag_to_tracks
from preprocessing.create_input_tracks import get_single_tracks

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path', help="path to rosbag folder")
    args = parser.parse_args()

    path = args.path 

    all_tracks = []

    counter = 0

    for file in os.listdir(path):
        if file.endswith(".bag"):
            counter += 1
            synced_data = rosbag_to_tracks(file, path)
            new_tracks = get_single_tracks(synced_data, counter)
            all_tracks.append(pd.DataFrame.from_dict(new_tracks))

    if len(all_tracks) > 1:
        df = pd.concat([all_tracks[0], all_tracks[1]], axis=0)
        if len(all_tracks) > 2:
            for i in all_tracks[2:]:
                df = pd.concat([df, i], axis=0)
    
    df.to_pickle('df.pkl')

