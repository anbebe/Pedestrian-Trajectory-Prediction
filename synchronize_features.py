import os
import pandas as pd
import numpy as np
import time
from bagpy import bagreader
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2

def msgs(bag):
    """For extracting msgs from Rosbag reader

    Returns:
        Dict[str, List]: list of messages and sequences of each kind
    """
    print("Extract messages")
    topic_list = [
        "/camera_left/color/image_raw",
        "/detected_persons/yolo",
        "/front_lidar/velodyne_points"
    ]
    image_msgs = []
    pc_msgs = []
    pers_msgs = []

    for topic, msg, t in bag.read_messages(topics=topic_list):
        
        if topic == "/camera_left/color/image_raw":
            image_msgs.append(msg)

        if topic =="/detected_persons/yolo":
            pers_msgs.append(msg)

        if topic == "/front_lidar/velodyne_points":
            pc_msgs.append(msg)


    bag.close()

    data_params = {
        "image_msgs": image_msgs,
        "pc_msgs": pc_msgs,
        "pers_msgs": pers_msgs,
    }

    return data_params

def create_detection_data(msgs):
    print("Get detection data")
    timestamps = []
    detections = []
    for i in msgs:
        timestamps.append(i.header.stamp.to_time())
        detection = []
        # reduce information about each detection to the necessary things
        for j in i.detections:
            # save from detections: confidence score, positionx, positiony, positionz, bboxx, bboxy, bboxw, bboxh
            conf_score = j.confidence
            pos = [j.pose.pose.position.x, j.pose.pose.position.y, j.pose.pose.position.z]
            bbox = [j.bbox_x, j.bbox_y, j.bbox_w, j.bbox_h]
            detection.append([conf_score, pos, bbox])
        detections.append(detection)
    data = {"timestamps": np.array(timestamps), "detections": detections}
    return data

def create_img_data(msgs):
    print("Get image data")
    timestamps = []
    img_msgs = []
    for i in msgs:
        timestamps.append(i.header.stamp.to_time())
        img_msgs.append(i)
    data = {"timestamps": np.array(timestamps), "img_msgs": img_msgs}
    return data

def pc_to_grid(tmp_msg):
    points_obj = pc2.read_points(tmp_msg, skip_nans=True, field_names=("x", "y", "z"))
    pc = np.array(list(points_obj), dtype=np.float32)
    pc[:,2] = 0
    #print(pc.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    #print(len(pcd.points))

    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center())
    #o3d.visualization.draw_geometries([pcd])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=0.01)
    #o3d.visualization.draw_geometries([voxel_grid])
    return voxel_grid

def create_vox_data(msgs):
    print("Get scene data")
    timestamps = []
    voxel_msgs = []
    for i in msgs:
        timestamps.append(i.header.stamp.to_time())
        vox = pc_to_grid(i)
        voxel_msgs.append(vox)
    data = {"timestamps": np.array(timestamps), "vox_msgs": voxel_msgs}
    return data

def sync_data(d1,d2,d3):
    print("Synchronize data")
    pd.set_option('display.float_format', '{:.2f}'.format)

    df_pers = pd.DataFrame.from_dict(data=d1)
    df_img = pd.DataFrame.from_dict(data=d2)
    df_vox = pd.DataFrame.from_dict(data=d3)

    df_pers.sort_values(by="timestamps",inplace=True)
    df_img.sort_values(by="timestamps",inplace=True)
    df_vox.sort_values(by="timestamps", inplace=True)

    merged_df = pd.merge_asof(df_pers, df_img, on="timestamps", direction="nearest")
    merged_df = pd.merge_asof(merged_df, df_vox, on="timestamps", direction="nearest")

    return merged_df

if __name__ == "__main__":

    start_time = time.time()

    path = os.path.join("/home/pbr-student/personal/thesis/crowdbot/rosbags_25_03_shared_control-rgbd_defaced", "defaced_2021-03-25-14-52-33.orig.bag")
    bag = bagreader(path).reader

    data_params = msgs(bag)

    detect_data = create_detection_data(data_params["pers_msgs"])
    img_data = create_img_data(data_params["image_msgs"])
    vox_data = create_vox_data(data_params["pc_msgs"])

    sync_data = sync_data(detect_data, img_data, vox_data)

    print("--- %s seconds ---" % (time.time() - start_time))