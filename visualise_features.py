import os
import sys
import pandas as pd
import cv2 as cv
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import tensorflow as tf
import imageio
import argparse

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_boxes(frame, detections):
    # Draw bounding boxes for tracked objects
    pose_frame = []
    for object in detections:
        if type(object[0])== int:
            x, y, w, h = object[2]
            x, y, w, h = int(x), int(y), int(w), int(h)
            object_id = object[0]
            confidence = object[-1]
            #pose_frame = draw_pose(frame, np.reshape(object[3], (1,1,17,3)))
            frame = cv.rectangle(frame.copy(), (x, y), (w, h), (0, 255, 0), 2)
            frame = cv.putText(frame, f"{str(object_id)}: {str(round(confidence, 2))}", (x, y - 10), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    return frame, pose_frame

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.

    Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        height: height of the image in pixels.
        width: width of the image in pixels.
        keypoint_threshold: minimum confidence score for a keypoint to be
        visualized.

    Returns:
        A (keypoints_xy, edges_xy, edge_colors) containing:
        * the coordinates of all keypoints of all detected entities;
        * the coordinates of all skeleton edges of all detected entities;
        * the colors in which the edges should be plotted.
    """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [np.array(kpts_x), np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
            kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[int(y_start), int(x_start)], [int(y_end), int(x_end)]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
        keypoints_xy = keypoints_xy.astype(int)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
    """Draws the keypoint predictions on image.

    Args:
        image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

    Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions.
    """
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin,ymin),rec_width,rec_height,
            linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image_from_plot = cv.resize(
            image_from_plot, dsize=(output_image_width, output_image_height),
            interpolation=cv.INTER_CUBIC)
    return image_from_plot


def gen_pose_gif(data, id):

    imgs = []

    for index, row in data.iterrows():
        img = row["img_msgs"].astype(np.uint8)
        for object in row["detections"]:
            if type(object[0])== int:
                if object[0] == id:
                    x, y, w, h = object[2]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cropped_img = img.copy()[y:h, x:w]
                    cropped_img = np.asarray(cropped_img.copy(),dtype=np.uint8)
                    cropped_img = draw_keypoints(cropped_img, object[3])
                    imgs.append(pad_img(cropped_img))

    imgs = np.asarray(imgs)


    imageio.mimsave('./animation.gif', imgs, duration=len(imgs))
    print("Saved gif as animation.gif")


def pad_img(im):
    desired_size = 368
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,
        value=color)
    
    return new_im


def draw_keypoints(frame, keypoints):

    keypoints_xy, edges_xy, edge_colors= _keypoints_and_edges_for_display(np.reshape(keypoints, (1,1,17,3)))

    for edge in edges_xy:
        frame = cv.line(frame,edge[0], edge[1],(255,0,0),2)
    return frame


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('path', help="path to data pickle file")
    parser.add_argument('--pose', help="gif for specific id poses should be generated", action="store_true")
    parser.add_argument('--pose_id', default=-1, help="which person should be tracked, if none chosen, random")
    parser.add_argument('--show-boxes', help="show detected bounding boxes and tracked ids", action="store_true")

    args = parser.parse_args()

    data = pd.read_pickle(args.path)

    if args.pose:
        detections = data.iloc[:]["detections"]
        all_ids = [x[0] for y in detections for x in y]
        occurs_id, occurs_counts = np.unique(np.asarray(all_ids), return_counts=True)
        if args.pose_id == -1:
            id = np.random.choice(occurs_id, 1)
        elif not np.isin(args.pose_id, occurs_id):
            raise ValueError("No valid person id, possible ids are: ", str(occurs_id))
        else:
            id = args.pose_id

        gen_pose_gif(data, id)

    if args.show_boxes:
        for index, row in data.iterrows():
            img = row["img_msgs"].astype(np.uint8)
            img, pose = draw_boxes(img, row["detections"])
            cv.imshow("frame", img)
            cv.waitKey(50)

        cv.destroyAllWindows()
    
    
