import os
import json
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm

def get_train_names():
    train = [
        "bytes-cafe-2019-02-07_0",
        "clark-center-2019-02-28_0",
        "clark-center-2019-02-28_1",
        "clark-center-intersection-2019-02-28_0",
        "cubberly-auditorium-2019-04-22_0",
        "forbes-cafe-2019-01-22_0",
        "gates-159-group-meeting-2019-04-03_0",
        "gates-ai-lab-2019-02-08_0",
        "gates-basement-elevators-2019-01-17_1",
        "gates-to-clark-2019-02-28_1",
        "hewlett-packard-intersection-2019-01-24_0",
        "huang-2-2019-01-25_0",
        "huang-basement-2019-01-25_0",
        "huang-lane-2019-02-12_0",
        "jordan-hall-2019-04-22_0",
        #"memorial-court-2019-03-16_0",
        "meyer-green-2019-03-16_0",
        "nvidia-aud-2019-04-18_0",
        "packard-poster-session-2019-03-20_0",
        "packard-poster-session-2019-03-20_1",
        "packard-poster-session-2019-03-20_2",
        "stlc-111-2019-04-19_0",
        "svl-meeting-gates-2-2019-04-08_0",
        #"svl-meeting-gates-2-2019-04-08_1",
        "tressider-2019-03-16_0",
        "tressider-2019-03-16_1",
        "tressider-2019-04-26_2",
    ]
    return train

# from human scene transformer, data subfolder
def get_file_handle(path, mode='rt'):
  file_handle = open(path, mode)
  return file_handle

def get_3d_features(input_path, scene):
    scene_data_file = get_file_handle(
        os.path.join(input_path, 'labels', 'labels_3d', scene + '.json')
    )

    # get agents_dict_from_detections
    scene_data = json.load(scene_data_file)
    agents = collections.defaultdict(list)
    for frame in scene_data['labels']:
        ts = int(frame.split('.')[0])
        for det in scene_data['labels'][frame]:
            agents[det['label_id']].append((ts, det))
    agents_dict = agents

    # get_agents_features_with_box
    max_distance_to_robot = 10
    agents_pos_dict = collections.defaultdict(dict)
    for agent_id, agent_data in agents_dict.items():
        for (ts, agent_instance) in agent_data:
            if agent_instance['attributes']['distance'] <= max_distance_to_robot:
                agents_pos_dict[(ts, agent_id)] = {
                    'timestamp:': ts,
                    'agent_id': agent_id,
                    'p': np.array([agent_instance['box']['cx'],
                                    agent_instance['box']['cy'],
                                    agent_instance['box']['cz']]),
                    # rotation angle is relative to negatiev x axis of robot
                    'yaw': np.pi - agent_instance['box']['rot_z'],
                    'l': agent_instance['box']['l'],
                    'w': agent_instance['box']['w'],
                    'h': agent_instance['box']['h']
                }
    agents_features = agents_pos_dict

    agents_df = pd.DataFrame.from_dict(
        agents_features, orient='index'
    ).rename_axis(['timestep', 'id']) 

    grouped_df2_1 = agents_df.groupby(agents_df['agent_id'], as_index=False)['timestamp:'].apply(list)
    grouped_df2_1 = grouped_df2_1.rename(columns={'agent_id': 'agent_id', 'timestamp:': 'timestep'})
    grouped_df2_2 = agents_df.groupby(agents_df['agent_id'], as_index=False)['p'].apply(list)
    grouped_result2 = pd.merge(grouped_df2_1, grouped_df2_2, on="agent_id")

    return grouped_result2

def get_keypoints(input_path, cam, scene):
    # get agents keypoints
    keypoint_path = os.path.join(input_path, 'labels', 'labels_2d_pose_coco')
    scene_data_file = get_file_handle(os.path.join(keypoint_path, scene + cam + '.json'))
    scene_data = json.load(scene_data_file)

    agents_keypoints = collections.defaultdict(dict)
    image_keypoints = collections.defaultdict(dict)

    for frame in scene_data['annotations']:
        agents_keypoints[frame['id']] = {
            'keypoints': np.array(frame['keypoints']).reshape(17, 3),
            'track_id': frame['track_id'],
            'timestep': frame['image_id']}
    for frame in scene_data['images']:
        image_keypoints[frame['id']] = {
            'timestep': frame['id'],
            'file_name': frame['file_name']
        }
    keypoints_df = pd.DataFrame.from_dict(
        agents_keypoints, orient='index'
    )
    image_df = pd.DataFrame.from_dict(
        image_keypoints, orient='index'
    )
    result = pd.merge(keypoints_df, image_df, on="timestep")

    grouped_df1 = result.groupby(result['track_id'], as_index=False)['timestep'].apply(list)
    grouped_df2 = result.groupby(result['track_id'], as_index=False)['file_name'].apply(list)
    grouped_df3 = result.groupby(result['track_id'], as_index=False)['keypoints'].apply(list)
    result12 = pd.merge(grouped_df1, grouped_df2, on="track_id")
    grouped_result = pd.merge(result12, grouped_df3, on="track_id")

    return grouped_result

def get_tracks(df1, df2):
    # merge 2d and 3d information together by tracks of timestamps
    track_dict = {}
    for idx in range(df1.shape[0]):
        tmp = df1.iloc[idx]['timestep']
        tmp_id = df1.iloc[idx]['track_id']
        tmp.sort()
        for index, row in df2.iterrows():
            if int(row['agent_id'][11:]) == int(tmp_id):
                #print(row['timestep'])
                # for the same track get the corresponding indices where the persons are detected at the same time
                indices1 = np.argwhere(np.isin(np.asarray(row['timestep']),np.asarray(tmp)))
                indices2 = np.argwhere(np.isin(np.asarray(tmp), np.asarray(row['timestep'])))
                # filter all relevant features by the indices
                if len(indices1) >= 15:
                    positions = np.asarray(row['p'])[indices1.squeeze()]
                    keypoints = np.asarray(df1.iloc[idx]['keypoints'])[indices2.squeeze()]
                    track_dict[idx] = {
                        'positions': positions,
                        'poses': keypoints
                    }
                break
    track_df = pd.DataFrame.from_dict(
        track_dict, orient='index'
    )
    return track_df

def get_train_pickle(train_scenes):
    for i in tqdm(range(len(train_scenes))):
        scene = train_scenes[i]
        print(scene)
        agents_df = get_3d_features(input_path, scene)
        for cam in ["_image0","_image2", "_image4", "_image6","_image8"]:
            keypoints_df = get_keypoints(input_path, cam, scene)
            track_df_tmp = get_tracks(keypoints_df, agents_df)
            all_tracks.append(track_df_tmp)

    if len(all_tracks) > 1:
        df = pd.concat([all_tracks[0], all_tracks[1]], axis=0)
        if len(all_tracks) > 2:
            for i in all_tracks[2:]:
                df = pd.concat([df, i], axis=0)
    print(df.shape[0])

    df.to_pickle('df_jrdb.pkl')

    return df

def get_augmented_train_pickle(df):
    df=df.reset_index(drop=True)
    def random_rotate(df):
        augmented_df = pd.DataFrame(columns=['positions', 'poses'])

        for i,row in df.iterrows():
            yaw = np.random.uniform(-np.pi, np.pi)
            
            # Create rotation matrix for the yaw angle
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rot_mat = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw,  cos_yaw, 0],
                [      0,        0, 1]
            ])

            augmented_df.loc[i] = [np.dot(row['positions'], rot_mat.T), np.dot(row['poses'], rot_mat.T)]
        return augmented_df

    def random_translate(df):
        augmented_df = pd.DataFrame(columns=['positions', 'poses'])

        for i,row in df.iterrows():
            translation = np.random.uniform(-10.0, 10.0, 2)
            translation = np.append(translation, [0.0])
            
            augmented_df.loc[i] = [row['positions'] + translation, row['poses']]

        return augmented_df

    new_df = random_rotate(df)
    new_df = random_translate(new_df)

    aug_df = pd.concat([df, new_df], ignore_index=True)

    aug_df.to_pickle('df_jrdb_augmented.pkl')

def get_test_pickle(test_scenes):
    for i in tqdm(range(len(test_scenes))):
        scene = test_scenes[i]
        print(scene)
        agents_df = get_3d_features(input_path, scene)
        for cam in ["_image0","_image2", "_image4", "_image6","_image8"]:
            keypoints_df = get_keypoints(input_path, cam, scene)
            track_df_tmp = get_tracks(keypoints_df, agents_df)
            all_tracks.append(track_df_tmp)

    if len(all_tracks) > 1:
        df = pd.concat([all_tracks[0], all_tracks[1]], axis=0)
        if len(all_tracks) > 2:
            for i in all_tracks[2:]:
                df = pd.concat([df, i], axis=0)
    print(df.shape[0])

    df.to_pickle('df_jrdb_test.pkl')



if __name__ == "__main__":
    input_path =  "/mnt/d/JRDB/train_dataset_with_activity/train_dataset_with_activity"
    train_scenes = get_train_names()

    all_tracks = []

    df = get_train_pickle(train_scenes)

    get_augmented_train_pickle(df)

    

