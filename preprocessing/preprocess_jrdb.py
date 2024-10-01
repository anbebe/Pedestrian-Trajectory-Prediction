"""
SCript to preprocess the raw rosbags from the JRDB dataset to a dataset of pedestrian trajectories with positions
and skeletal keypoints.
many parts taken from JRDB preprocessing of the HST https://github.com/google-research/human-scene-transformer
"""
import os
import json
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_math import pose3, rotation3, quaternion

def get_train_names():
    """
    Returns a list of predefined training scenes (about 80 percent of the whole dataset), from HST implementation

    :returns list of string
    """
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
        "meyer-green-2019-03-16_0",
        "nvidia-aud-2019-04-18_0",
        "packard-poster-session-2019-03-20_0",
        "packard-poster-session-2019-03-20_1",
        "packard-poster-session-2019-03-20_2",
        "stlc-111-2019-04-19_0",
        "svl-meeting-gates-2-2019-04-08_0",
        "tressider-2019-03-16_0",
        "tressider-2019-03-16_1",
        "tressider-2019-04-26_2",
    ]
    return train

# from human scene transformer, data subfolder
def get_file_handle(path, mode='rt'):
  """
  Opens a file at the given math with read text
  """
  file_handle = open(path, mode)
  return file_handle

def get_3d_features(input_path, scene):
    """
    Extracts 3D features pf agents in a given scene

    :param input_path: path to data directory
    :param scene: name of the scene

    :returns pandas dataframe containing the agents features (3d position to corresponding agent id)
    """
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

    level = agents_df.index.levels[0]

    # group features for each agent
    grouped_df2_1 = agents_df.groupby(agents_df['agent_id'], as_index=False)['timestamp:'].apply(list)
    grouped_df2_1 = grouped_df2_1.rename(columns={'agent_id': 'agent_id', 'timestamp:': 'timestep'})
    grouped_df2_2 = agents_df.groupby(agents_df['agent_id'], as_index=False)['p'].apply(list)
    grouped_df2_3 = agents_df.groupby(agents_df['agent_id'], as_index=False)['yaw'].apply(list)
    # merge grouped results
    grouped_result2 = pd.merge(grouped_df2_1, grouped_df2_2, on="agent_id")
    grouped_result2 = pd.merge(grouped_result2, grouped_df2_3, on="agent_id")

    return grouped_result2, level

def get_keypoints(input_path, cam, scene):
    """
    Extracts skeletal keypoints from given camera

    :param input_path: path to data directory
    :param cam: cmaera identifier (name of one of the five cameras)
    :param scene: name of the scene

    :returns pandas dataframe containing the skeletal keypoins corresponding to its agents id
    """
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
    # merge keypoints and images
    result = pd.merge(keypoints_df, image_df, on="timestep")

    # group by track id
    grouped_df1 = result.groupby(result['track_id'], as_index=False)['timestep'].apply(list)
    grouped_df2 = result.groupby(result['track_id'], as_index=False)['file_name'].apply(list)
    grouped_df3 = result.groupby(result['track_id'], as_index=False)['keypoints'].apply(list)
    # merge grouped results
    result12 = pd.merge(grouped_df1, grouped_df2, on="track_id")
    grouped_result = pd.merge(result12, grouped_df3, on="track_id")

    return grouped_result

def get_keypoints_odom(input_path, cam, scene, robot_df):
    """
    Extracts skeletal keypoints from given camera and merges with odometry data

    :param input_path: path to data directory
    :param cam: cmaera identifier (name of one of the five cameras)
    :param scene: name of the scene
    :param robot_df: dataframe with odometry data by timestamp sorted for that scene

    :returns pandas dataframe containing the skeletal keypoins corresponding to its agents id and the odometry data
    """
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
    result = pd.merge(result, robot_df, on="timestep")

    # group by track id
    grouped_df1 = result.groupby(result['track_id'], as_index=False)['timestep'].apply(list)
    grouped_df2 = result.groupby(result['track_id'], as_index=False)['file_name'].apply(list)
    grouped_df3 = result.groupby(result['track_id'], as_index=False)['keypoints'].apply(list)
    grouped_df4 = result.groupby(result['track_id'], as_index=False)['pq'].apply(list)
    # merge grouped results
    result12 = pd.merge(grouped_df1, grouped_df2, on="track_id")
    grouped_result = pd.merge(result12, grouped_df3, on="track_id")
    grouped_result = pd.merge(grouped_result, grouped_df4, on="track_id")

    return grouped_result

def get_tracks(df1, df2):
    """
    Merge 2d and 3d information together by tracks of timestamps via combining the two dataframes containing
    each respectively the keypoints and positions.

    :param df1: pandas dataframe containing skeletal keypoints sorted by track id and with timestamp
    :param df2: pandas dataframe containing positions sorted by track id and with timestamp

    :returns pandas dataframe where each row corresponds to a matched track containing positions and skeletal keypoints
    """
    track_dict = {}
    # Iterate through position dataframe
    for idx in range(df1.shape[0]):
        tmp = df1.iloc[idx]['timestep']
        tmp_id = df1.iloc[idx]['track_id']
        tmp.sort()
        # iterate trough pose dataframe and search for matching agents
        for index, row in df2.iterrows():
            # check if agents match in both rows
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

def get_tracks_odom(df1, df2):
    """
    Merge 2d and 3d information together by tracks of timestamps via combining the two dataframes containing
    each respectively the keypoints and positions. Additionally get the yaw and odom data for later transformation,

    :param df1: pandas dataframe containing skeletal keypoints sorted by track id and with timestamp
    :param df2: pandas dataframe containing positions sorted by track id and with timestamp

    :returns pandas dataframe where each row corresponds to a matched track containing positions and skeletal keypoints
    """
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
                    yaw = np.asarray(row['yaw'])[indices1.squeeze()]
                    keypoints = np.asarray(df1.iloc[idx]['keypoints'])[indices2.squeeze()]
                    odom = np.asarray(df1.iloc[idx]['pq'])[indices2.squeeze()]
                    track_dict[idx] = {
                        'positions': positions,
                        'yaw': yaw,
                        'poses': keypoints,
                        'odom': odom
                    }
                break
    track_df = pd.DataFrame.from_dict(
        track_dict, orient='index'
    )
    return track_df

def get_robot_odom(input_path, scene, level):
    """Returns robot features from raw data."""
    odom_data_file = os.path.join(input_path, 'odometry', 'train')
    odom_data_file = get_file_handle(
        os.path.join(odom_data_file, scene + '.json'))
    odom_data = json.load(odom_data_file)

    robot = collections.defaultdict(list)

    for pc_ts, pose in odom_data['odometry'].items():
        ts = int(pc_ts.split('.')[0])
        robot[ts] = {
            'p': np.array([pose['position']['x'],
                        pose['position']['y'],
                        pose['position']['z']]),
            'q': np.array([pose['orientation']['x'],
                        pose['orientation']['y'],
                        pose['orientation']['z'],
                        pose['orientation']['w']]),
        }
    robot_df = pd.DataFrame.from_dict(robot, orient='index').rename_axis(  
        ['timestep']
    )
    # Remove extra data odometry datapoints
    robot_df = robot_df.iloc[level]
    robot_df['pq'] = robot_df['p'].apply(list) + robot_df['q'].apply(list)
    return robot_df


def get_train_pickle(train_scenes):
    """
    Processes each scene by extracting the reelvant features and merging the extracted trajectories 
    in one dataframe thatt is saved lcoally as pickle.

    :param train_scenes: list of scenes

    :returns a pandas dataframe containing all processed tracks with postion and pose information
    """
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
    """
    Augments the given dataframe by applying random rotation and translation to positon and skeletal keypoints from odometry data.

    :param df: pandas dataframe containing tracks with positional and skeletal keypoint information
    """
    df=df.reset_index(drop=True)
    def random_rotate(df):
        """
        Randomly rotates the psotions and poses around the z-axis (yaw)

        :param df: pandas dataframe containing tracks with positional and skeletal keypoint information

        :returns a new pandas dataframe with rotated features
        """
        augmented_df = pd.DataFrame(columns=['positions', 'poses'])

        for i,row in df.iterrows():
            yaw = np.random.uniform(-np.pi/4, np.pi/4)
            
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
        """
        Randomly translates the psotions and poses in x and y direction (ground plane)

        :param df: pandas dataframe containing tracks with positional and skeletal keypoint information

        :returns a new pandas dataframe with translated features
        """
        augmented_df = pd.DataFrame(columns=['positions', 'poses'])

        for i,row in df.iterrows():
            translation = np.random.uniform(-1.0, 1.0, 2)
            translation = np.append(translation, [0.0])
            
            augmented_df.loc[i] = [row['positions'] + translation, row['poses']]

        return augmented_df

    new_df = random_rotate(df)
    new_df = random_translate(new_df)

    aug_df = pd.concat([df, new_df], ignore_index=True)

    aug_df.to_pickle('df_jrdb_odom_augmented.pkl')
    

def get_test_pickle(test_scenes):
    """
    Processes each scene by extracting the reelvant features and merging the extracted trajectories 
    in one dataframe thatt is saved lcoally as pickle.

    :param test_scenes: list of scenes

    :returns a pandas dataframe containing all processed tracks with postion and pose information
    """
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


def agents_to_odom_frame(robot_df, track_df):
    """
    Transforms agent positions and skeletal keypoints from the world frame to the robot's odometry frame

    :param robot_df: pandas dataframe containing the robots 6DOF pose in the world frame
    :param track_df: pandas dataframe containing agents tracks with positions, poses and orientation (yaw)

    :returns new pandas datafrae with tansformd positions and keypoints
    """
    # robots pose at first timestep is used as initial transformation from world to robt frame
    world_pose_odometry = pose3.Pose3(
        rotation3.Rotation3(
            quaternion.Quaternion(robot_df.loc[0]['q'])), robot_df.loc[0]['p'])
    # inverse gives the transformation from odom to world data
    odometry_pose_world = world_pose_odometry.inverse()

    agents_dict = {}
    # iterate over tracks
    for index, row in track_df.iterrows():
        pos = []
        pose = []
        # iterate over single steps of trajectory
        for i in range(len(row["positions"])):
            # get position p and orientation q of robot at that timestep
            robot_odometry_dp = {
                        'p': row["odom"][i,:3],
                        'q': row["odom"][i,3:]
                    }
            # get robots pose at that timestep
            world_pose_robot = pose3.Pose3(
                rotation3.Rotation3(
                    quaternion.Quaternion(robot_odometry_dp['q'])),
                robot_odometry_dp['p'])
            # get agents pose at that timestep
            robot_pose_agent = pose3.Pose3(
                rotation3.Rotation3.from_euler_angles(
                    rpy_radians=[0., 0., row['yaw'][i]]), row['positions'][i])
            # transform agents pose from local robot frame to odometry frame
            odometry_pose_agent = (odometry_pose_world * world_pose_robot
                            * robot_pose_agent)
            pos.append(odometry_pose_agent.translation)
            # compute rotation from world to robots frame
            world_rot_robot = rotation3.Rotation3(
            quaternion.Quaternion(robot_odometry_dp['q']))
            # only rotate the keypoints (not translate since in image space)
            odometry_rot_robot = odometry_pose_world.rotation * world_rot_robot
            rot_keypoints = []
            for keypoint in row['poses'][i]:
                if np.isnan(keypoint).any():
                    rot_keypoints.append(keypoint)
                else:
                    rot_keypoints.append(odometry_rot_robot.rotate_point(keypoint))
            rot_keypoints = np.array(rot_keypoints)
            pose.append(rot_keypoints)
        if index not in agents_dict:
            agents_dict[index] = {}
        agents_dict[index]['positions'] = np.asarray(pos)
        agents_dict[index]['poses'] = np.asarray(pose)
    odom_track_df = pd.DataFrame.from_dict(
        agents_dict, orient='index'
    )
    
    return odom_track_df



def get_odom_train_pickle(train_scenes):
    """
    Processes each scene by extracting the reelvant features and merging the extracted trajectories 
    and transferring the features from local robot to world frame in one dataframe that is saved locally as pickle.

    :param train_scenes: list of scenes

    :returns a pandas dataframe containing all processed tracks with postion and pose information
    """
    for i in tqdm(range(len(train_scenes))):
        scene = train_scenes[i]
        print(scene)
        agents_df, level = get_3d_features(input_path, scene)
        for cam in ["_image0","_image2", "_image4", "_image6","_image8"]:
            odom_df = get_robot_odom(input_path,scene, level)
            keypoints_df = get_keypoints_odom(input_path, cam, scene, odom_df)
            track_df_tmp = get_tracks_odom(keypoints_df, agents_df)
            track_df_odom = agents_to_odom_frame(odom_df, track_df_tmp)
            all_tracks.append(track_df_odom)

    if len(all_tracks) > 1:
        df = pd.concat([all_tracks[0], all_tracks[1]], axis=0)
        if len(all_tracks) > 2:
            for i in all_tracks[2:]:
                df = pd.concat([df, i], axis=0)
    print(df.shape[0])

    df.to_pickle('df_jrdb_odom.pkl')
    return df


if __name__ == "__main__":
    input_path =  "/mnt/d/JRDB/train_dataset_with_activity/train_dataset_with_activity"
    train_scenes = get_train_names()

    all_tracks = []

    df = get_odom_train_pickle(train_scenes)

    get_augmented_train_pickle(df)

    

