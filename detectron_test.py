import os

import numpy as np
import matplotlib.pyplot as plt
from bagpy import bagreader
from rosbags.image import message_to_cvimage
from collections import defaultdict
import cv2 as cv
import open3d as o3d
import detectron2
import sensor_msgs.point_cloud2 as pc2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import pyrealsense2 as rs2
if (not hasattr(rs2, 'intrinsics')):
    import pyrealsense2.pyrealsense2 as rs2


def msgs(bag):
    """For extracting msgs from Rosbag reader

    Returns:
        Dict[str, List]: list of messages and sequences of each kind
    """
    topic_list = [
        "/camera_left/color/image_raw",
        "/camera_left/aligned_depth_to_color/image_raw",
        "/camera_left/depth/color/points",
        "/camera_left/aligned_depth_to_color/camera_info",
        "/detected_persons/yolo"
    ]
    image_msgs = []
    depth_msgs = []
    depth_pts_msgs = []
    pers_msgs = []
    ci_msgs = []

    counter1 = 100
    counter2 = 100
    counter3 = 100
    counter4 = 100
    counter5 = 100

    for topic, msg, t in bag.read_messages(topics=topic_list):
        
        if counter1 > 0 or counter2 > 0 or counter3 > 0 or counter4 > 0 or counter5>0:
            if topic == "/camera_left/color/image_raw":
                image_msgs.append(msg)
                counter1 -= 1

            if topic == "/camera_left/aligned_depth_to_color/image_raw":
                depth_msgs.append(msg)
                counter2 -= 1

            if topic == "/camera_left/depth/color/points":
                depth_pts_msgs.append(msg)
                counter3 -= 1

            if topic == "/camera_left/aligned_depth_to_color/camera_info":
                ci_msgs.append(msg)
                counter4 -= 1

            if topic =="/detected_persons/yolo":
                pers_msgs.append(msg)
                counter5 -=1
        else:
                break


    bag.close()

    data_params = {
        "image_msgs": image_msgs,
        "depth_msgs": depth_msgs,
        "depth_pts_msgs": depth_pts_msgs,
        "pers_msgs": pers_msgs,
        "ci_msgs": ci_msgs,
    }

    return data_params

def get_pointclouds(data_params):
    pcs = []
    for i in [80]:
        tmp_msg = data_params["pc_msgs"][i]
        points_obj = pc2.read_points(tmp_msg, skip_nans=True, field_names=("x", "y", "z"))
        pc = np.array(list(points_obj), dtype=np.float32)
        print(pc.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.io.write_point_cloud("sync2.ply", pcd)

        # Load saved point cloud and visualize it
        pcd_load = o3d.io.read_point_cloud("sync2.ply")
        o3d.visualization.draw_geometries([pcd_load])

        pcs.append(pc)
        break

    return pcs

def get_frames(data_params):
    imgs = []
    depths = []
    for i in range(100):
        tmp_msg = data_params["image_msgs"][i]
        img = message_to_cvimage(tmp_msg)
        #print("img header: ",tmp_msg.header)
        imgs.append(img)

        depth_msg = data_params["depth_msgs"][i]
        img2 = message_to_cvimage(depth_msg)
        depths.append(img2)

        break
    
    return imgs, depths

def get_camera_info(data_params):
    tmp_msg = data_params["ci_msgs"][0]
    return tmp_msg

def load_model():
    # Load the model
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    return cfg, predictor

def track(images):
    cfg, predictor = load_model()
    # Store the track history
    track_history = defaultdict(lambda: [])
    print(images[0].shape)
    height,width,layers= (576,768,3)#images[0].shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v') 
    video = cv.VideoWriter('video2.mp4', fourcc, 1, (width, height))

    # Loop through the video frames
    for frame in images:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        outputs = predictor(frame)
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        annotated_frame = out.get_image()[:, :, ::-1]
        print(annotated_frame.shape)
        cv.imshow("test", annotated_frame)
        cv.waitKey(0)
        break
        video.write(annotated_frame)
        
    cv.destroyAllWindows()
    video.release()


def get_positions(images, depths, cameraInfo):

    results = []

    img = images[0].copy()
    depth = depths[0]

    # load intrinsic parameters for depth calculation
    intrinsics = rs2.intrinsics()
    intrinsics.width = cameraInfo.width
    intrinsics.height = cameraInfo.height
    intrinsics.ppx = cameraInfo.K[2]
    intrinsics.ppy = cameraInfo.K[5]
    intrinsics.fx = cameraInfo.K[0]
    intrinsics.fy = cameraInfo.K[4]
    if cameraInfo.distortion_model == 'plumb_bob':
        intrinsics.model = rs2.distortion.brown_conrady
    elif cameraInfo.distortion_model == 'equidistant':
        intrinsics.model = rs2.distortion.kannala_brandt4
    intrinsics.coeffs = [i for i in cameraInfo.D] 

    cfg, predictor = load_model()
    outputs = predictor(img)

    for i in range(len(outputs['instances'].get('pred_boxes'))):
        bbox = outputs['instances'].get('pred_boxes')[i].to("cpu")
        score = outputs['instances'].get('scores')[i].to("cpu")
        center = bbox.get_centers().numpy()[0]
        bbox_np = bbox.tensor.numpy()[0]
        
        if center[0] < img.shape[0] and center[1] < img.shape[1] and score > 0.9:
            depth_val = depth[int(center[0])][int(center[1])]

            img = cv.rectangle(img, (int(bbox_np[0]), int(bbox_np[1])), (int(bbox_np[2]), int(bbox_np[3])), (0,255,0), 3)

            img =  cv.circle(img, (int(center[0]), int(center[1])), radius=0, color=(0,0,255), thickness=4)

            result = rs2.rs2_deproject_pixel_to_point(intrinsics, [int(bbox_np[0]), int(bbox_np[1])], depth_val)
            print(result)
            results.append(result)

    cv.imwrite("detected.jpg", img)
    return np.asarray(results)
    

def get_yolo_position(data_params):
    positions = []
    ids = []
    for i in range(100):
        tmp_msg = data_params["pers_msgs"][i]
        #print("detected persons header: ", tmp_msg.header)
        tmp_ids = []
        tmp_pos = []
        for i in tmp_msg.detections:
            tmp_ids.append(i.detection_id)
            tmp_pos.append([i.pose.pose.position.x, i.pose.pose.position.y, i.pose.pose.position.z])

        break
    print(np.asarray(tmp_pos))
    return np.asarray(tmp_pos)

def visualise_pos(pos, y_pos):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(projection='3d')

    ax.scatter(pos[:,0], pos[:,1], pos[:,2], marker='o')
    ax2.scatter(y_pos[:,0], y_pos[:,1], y_pos[:,2], marker='^')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')


    plt.show()
    



if __name__ == "__main__":

    path = os.path.join("/home/pbr-student/personal/thesis/crowdbot/rosbags_25_03_shared_control-rgbd_defaced", "defaced_2021-03-25-14-52-33.bag")
    bag = bagreader(path).reader

    data_params = msgs(bag)

    yolo_pos = get_yolo_position(data_params)

    #pcs = get_pointclouds(data_params)

    imgs, depths = get_frames(data_params)
    #track(imgs)
    cameraInfo = get_camera_info(data_params)
    pos = get_positions(imgs, depths, cameraInfo)

    #visualise_pos(pos, yolo_pos)