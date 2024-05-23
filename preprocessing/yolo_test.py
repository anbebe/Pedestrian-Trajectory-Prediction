import os

import numpy as np

from bagpy import bagreader
import os
from rosbags.image import message_to_cvimage
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2 as cv

from ultralytics import YOLO

def msgs(bag):
    """For extracting msgs from Rosbag reader

    Returns:
        Dict[str, List]: list of messages and sequences of each kind
    """
    topic_list = [
        "/camera_left/color/image_raw",
        "/camera_left/aligned_depth_to_color/image_raw",
        "/camera_left/depth/color/points",
        "/front_lidar/scan"
    ]
    image_msgs = []
    depth_msgs = []
    depth_pts_msgs = []
    pc_msgs = []

    counter1 = 100
    counter2 = 100
    counter3 = 100
    counter4 = 100

    for topic, msg, t in bag.read_messages(topics=topic_list):
        
        if counter1 > 0 or counter2 > 0 or counter3 > 0 or counter4 > 0:
            if topic == "/camera_left/color/image_raw":
                image_msgs.append(msg)
                counter1 -= 1

            if topic == "/camera_left/aligned_depth_to_color/image_raw":
                depth_msgs.append(msg)
                counter2 -= 1

            if topic == "/camera_left/depth/color/points":
                depth_pts_msgs.append(msg)
                counter3 -= 1

            if topic == "/front_lidar/scan":
                pc_msgs.append(msg)
                counter4 -= 1
        else:
                break


    bag.close()

    data_params = {
        "image_msgs": image_msgs,
        "depth_msgs": depth_msgs,
        "depth_pts_msgs": depth_pts_msgs,
        "pc_msgs": pc_msgs
    }

    return data_params

def get_frames(data_params):
    imgs = []
    depths = []
    for i in range(100):
        tmp_msg = data_params["image_msgs"][i]
        img = message_to_cvimage(tmp_msg)
        imgs.append(img)

        depth_msg = data_params["depth_msgs"][i]
        img2 = message_to_cvimage(depth_msg)
        depths.append(img2)
    return imgs, depths

def track(images):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    # Store the track history
    track_history = defaultdict(lambda: [])

    height,width,layers=images[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video = cv.VideoWriter('video.mp4', fourcc, 1, (width, height))

    # Loop through the video frames
    for frame in images:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):

            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        video.write(annotated_frame)

    cv.destroyAllWindows()
    video.release()


    



if __name__ == "__main__":

    path = os.path.join("/home/annalena/crowdbot-evaluation-tools/data/rosbags_03_12_manual-rgbd_defaced/", "defaced_2021-12-03-19-12-00.bag")
    bag = bagreader(path).reader

    data_params = msgs(bag)
    imgs, depths = get_frames(data_params)
    track(imgs)