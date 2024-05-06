import os
import pandas as pd
import numpy as np
import time
from bagpy import bagreader
import open3d as o3d
import cv2 as cv
from motpy import Detection, MultiObjectTracker
import numpy as np
import pandas as pd
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm
import sensor_msgs.point_cloud2 as pc2
from rosbags.image import message_to_cvimage
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2
pose_sample_rpi_path = os.path.join(os.getcwd(), 'examples/lite/examples/pose_estimation/raspberry_pi')
sys.path.append(pose_sample_rpi_path)
import utils
from ml import Movenet

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
    """ loads timestamps and corresponding detection data (confidence in detected bounding box, bounding box (x,y,w,h)
    and 3d position) for possibly mutliple persons in one image

    Args:
        msgs: message from rosbag

    Returns: 
        dictionary with timestamps and detection data 
        for each detected person: (confidence score, positionx, positiony, positionz, bboxx, bboxy, bboxw, bboxh)
    """
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
    """loads timestamps and corresponding images from message, converts images to usable matrixes

    Args:
        msgs: message from rosbag

    Returns:
        dictionary with timestamps and image data
    """
    print("Get image data")
    timestamps = []
    img_msgs = []
    for i in msgs:
        timestamps.append(i.header.stamp.to_time())
        img = message_to_cvimage(i)
        img_msgs.append(img)
    data = {"timestamps": np.array(timestamps), "img_msgs": img_msgs}
    return data

def pc_to_grid(tmp_msg):
    """Loads points from rosbag message, ignores the z-axes to get a 2d grid and creates a VoxelGrid

    Args:
        tmp_msg: message

    Returns:
        indices of VoxelGrid voxels
    """
    points_obj = pc2.read_points(tmp_msg, skip_nans=True, field_names=("x", "y", "z"))
    pc = np.array(list(points_obj), dtype=np.float32)
    # reduces points to 2 dimensions
    pc[:,2] = 0 

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # fit to unit cube
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center())
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=0.01)
    voxels = voxel_grid.get_voxels()  # returns list of voxels
    indices = np.stack(list(vx.grid_index for vx in voxels))
    #voxel_coord = np.stack(list(voxel_grid.get_voxel_center_coordinate(vx) for vx in indices))
    indices = np.stack(list(vx.grid_index for vx in voxels))[:,:2]

    return indices

def create_vox_data(msgs):
    """loads timestamps and corresponding pointcloud data, reduces pointclouds to 2d voxelgrids 

    Args:
        msgs: message from rosbag

    Returns:
        returns dictionary with timestamps and voxelgrid data
    """
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
    """Synchronize three datasets (dictionaries) by the timestamp key

    Args:
        d1: detection dataset
        d2: image dataset
        d3: scene dataset

    Returns:
        one dataframe with combined datasets by timestamo
    """
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

def draw_boxes(frame, track_results, id_dict):
    # Draw bounding boxes for tracked objects
    for object in track_results:
        #print("object: ", object)
        x, y, w, h = object.box
        x, y, w, h = int(x), int(y), int(w), int(h)
        object_id = object.id
        confidence = object.score
        cv.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv.putText(frame, f"{str(id_dict[object_id])}: {str(round(confidence[0], 2))}", (x, y - 10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv.putText(frame, "People Count: {}".format(len(track_results)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def update_id_dict(id_dict, j, track_results):
    # Update the dictionary with new object IDs and corresponding labels
    for track_result in track_results:
        if track_result.id not in id_dict:
            id_dict[track_result.id] = j
            j += 1
    return id_dict, j

def detect(movenet, input_tensor, inference_count=3):
    """Runs detection on an input image.
    
    Args:
        input_tensor: A [height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.
        inference_count: Number of times the model should run repeatly on the
        same input image to improve detection accuracy.
    
    Returns:
        A Person entity detected by the MoveNet.SinglePose.
    """
    image_height, image_width, channel = input_tensor.shape
    
    # Detect pose using the full input image
    movenet.detect(input_tensor, reset_crop_region=True)
    
    # Repeatedly using previous detection result to identify the region of
    # interest and only croping that region to improve detection accuracy
    for _ in range(inference_count - 1):
        person = movenet.detect(input_tensor, 
                                reset_crop_region=False)

    return person

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
    """Draws the keypoint predictions on image.

    Args:
    image: An numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
    person: A person entity returned from the MoveNet.SinglePose model.
    close_figure: Whether to close the plt figure after the function returns.
    keep_input_size: Whether to keep the size of the input image.

    Returns:
    An numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
    """
    # Draw the detection result on top of the image.
    image_np = utils.visualize(image, [person])

    # Plot the image with detection results.
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    im = ax.imshow(image_np)

    if close_figure:
        plt.close(fig)

    if not keep_input_size:
        image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

    return image_np

def get_pose2(movenet, frame, bbox):
    #offset = 0
    #cropped_img = frame.copy()[bbox[1]-offset:bbox[3]+offset, bbox[0]-offset:bbox[2]+offset]
    cropped_img = frame.copy()[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    cropped_img = np.asarray(cropped_img.copy(),dtype=np.uint8)
    #cropped_img = frame.copy()
    person = detect(movenet, cropped_img)

    #print("detection result: ", person)
    detections = []
    if len(person.keypoints) > 0:
        output_overlay = draw_prediction_on_image(
              cropped_img.astype(np.uint8), person, 
              close_figure=True, keep_input_size=True)
        #output_frame = cv.cvtColor(output_overlay, cv.COLOR_RGB2BGR)
        #plt.imshow(output_frame)
        for keypoint in person.keypoints:
            detections.append([keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score])
    return detections

def update_dets(idx, track_res, id_dict, test_syn_data, movenet):
    new_sync_data = test_syn_data
    for i in range(len(test_syn_data["detections"][idx])):
        obj = test_syn_data["detections"][idx][i]
        conf, pos, bbox_orig = obj
        x_min, y_min, w, h = bbox_orig
        bbox_orig = np.array([int(x_min), int(y_min), int(x_min+w), int(y_min+h)])
        # get bestimmt effizienter
        for obj_ in track_res:
            track_box = obj_.box.astype(int)
            if np.all(np.abs(bbox_orig - track_box) < 15 ):
                # get pose from blazepose
                #print(np.abs(bbox_orig - track_box))
                poses = get_pose2(movenet, test_syn_data["img_msgs"][idx], bbox_orig)
                id = id_dict[obj_.id]
                new_sync_data["detections"][idx][i] = [id, pos, bbox_orig, poses, conf]
                break
    return new_sync_data

def track(sync_data, movenet):
    """Uses Multiobject Tracker and Movenet to update detection information through all timestamps;
    Tracks persons with unique ids by given bounding boxes and detects keypoints poses with movenet on
    cropped images (from bounding boxes)

    Args:
        sync_data: pandas dataframe with synchronized image, detection and scene data
        movenet: pretrained and instantiated movenet model

    Returns:
        updated dictionary (no pandas dataframe) 
    """
    test_syn_data = sync_data.copy().to_dict()
    imgs = sync_data.iloc[:]["img_msgs"]

    bboxes = []
    scores = []
    for d in sync_data.iloc[:]["detections"]:
        bbox_d = []
        scores_d = []
        for i in d:
            x_min, y_min, w, h = i[2]
            bbox_d.append([int(x_min), int(y_min), int(x_min+w), int(y_min+h)])
            scores_d.append([i[0]])
        bboxes.append(bbox_d)
        scores.append(scores_d)


    # Initialize MultiObjectTracker
    tracker = MultiObjectTracker(dt=1 / 15, tracker_kwargs={'max_staleness': 10})

    # Initialize ID dictionary and counter
    id_dict = {}
    j = 0

    for frame_id in tqdm(range(len(imgs))):

        detections = []

        # Pass YOLO detections to motpy
        for coord, score in zip(bboxes[frame_id], scores[frame_id]):
            detections.append(Detection(box=coord, score=score, class_id=25))

        # Perform object tracking
        tracker.step(detections=detections)
        track_results = tracker.active_tracks()

        # Update ID dictionary
        id_dict, j = update_id_dict(id_dict, j, track_results)

        test_syn_data = update_dets(frame_id, track_results, id_dict, test_syn_data, movenet)


    return test_syn_data

def draw_boxes(frame, detections):
    # Draw bounding boxes for tracked objects
    for object in detections:
        if type(object[0])== int:
            x, y, w, h = object[2]
            x, y, w, h = int(x), int(y), int(w), int(h)
            object_id = object[0]
            confidence = object[-1]
            frame = cv.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            frame = cv.putText(frame, f"{str(object_id)}: {str(round(confidence, 2))}", (x, y - 10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return frame

def filter_detections_id_pose(df, index, row):
    """filter trajectories with enough detected keypoints and length of sequence

    Args:
        df: synchronized dataframe
        index: index of row
        row: row of dataset

    Returns:
        updated dataframe
    """
    old_detections = row["detections"]
    new_detections = []
    for object in old_detections:
        if type(object[0])== int:
            object_id = object[0]
            confidence = object[-1]
            pose_kps = object[3]
            pose_confs = np.asarray(pose_kps)[:,-1]
            # if half of the keypoints are detected with enough confidence, the person is kept
            if len(np.argwhere(pose_confs > 0.5)) > 8 and not type(confidence) == list:
                if confidence > 0.7:
                    new_detections.append([object_id, object[1], object[2], pose_kps, object[4]])
    df.update(pd.DataFrame({'detections': [new_detections]}, index=[index]))
    return df

def filter_seq_len(df):
    """Get all persons (ids) that occur at least 10 times in the sequence

    Args:
        df: synchronized dataframe

    Returns:
        ids to keep
    """
    ids = []
    detections = df.iloc[:]["detections"]
    all_ids = [x[0] for y in detections for x in y]
    occurs_id, occurs_counts = np.unique(np.asarray(all_ids), return_counts=True)
    occurs_idx = np.argwhere(occurs_counts>20) # TODO: parameter, gucken wie viel übrig bleibt
    ids = occurs_id[occurs_idx]
    return np.squeeze(ids)

def filter_ids(df, row, ids):
    """Filters the dataframe, to keep only detections of persons with specific ids

    Args:
        df: synchronized dataframe
        row: row 
        ids: ids to keep 

    Returns:
        updated dataframe
    """
    old_detections = row["detections"]
    new_detections = []
    for object in old_detections:
        if np.isin(object[0], ids):
            object_id = object[0]
            confidence = object[-1]
            pose_kps = object[3]
            pose_confs = np.asarray(pose_kps)[:,-1]
            new_detections.append([object_id, object[1], object[2], pose_kps, object[4]])
    df.update(pd.DataFrame({'detections': [new_detections]}, index=[index]))
    return df


if __name__ == "__main__":

    start_time = time.time()
    path = os.path.join("/home/pbr-student/personal/thesis/crowdbot/rosbags_10_04_mds-rgbd_defaced", "defaced_2021-04-10-11-56-56-001.bag")

    bag = bagreader(path).reader

    data_params = msgs(bag)

    detect_data = create_detection_data(data_params["pers_msgs"])
    img_data = create_img_data(data_params["image_msgs"])
    vox_data = create_vox_data(data_params["pc_msgs"])

    sync_data = sync_data(detect_data, img_data, vox_data)
    print("--- %s seconds ---" % (time.time() - start_time))

    # track persons and add id and pose
    
    movenet = Movenet('pretrained/movenet_thunder')

    full_data = track(sync_data, movenet)
    print("--- %s seconds ---" % (time.time() - start_time))

    full_data = pd.DataFrame.from_dict(full_data)

    # filter trajectories by length and confidence in keypoints 

    for index, row in full_data.copy().iterrows():
        full_data = filter_detections_id_pose(full_data, index, row)

    keep_ids = filter_seq_len(full_data)

    for index, row in full_data.iterrows():
        full_data = filter_ids(full_data, row, keep_ids)

    # save data as pickle file    """

    sync_data.to_pickle('synced_full_data.pkl')



