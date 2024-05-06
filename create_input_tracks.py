from tqdm import tqdm
import pandas as pd
import cv2 as cv
import numpy as np
import imageio
from skimage.metrics import mean_squared_error as mse
from scipy import spatial
import torchreid
import pandas as pd

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

def get_cropped_img(bbox, img):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    cropped_img = img.copy()[y:h, x:w]
    cropped_img = np.asarray(cropped_img.copy(),dtype=np.uint8)
    return cropped_img

def pad_voxelgrids(voxel_list):
    voxels = []
    for v in voxel_list:
        v = np.asarray(v)
        pad_size = 1000 - v.shape[0]

        if pad_size > 0:
            v = np.pad(v,((0,pad_size),(0,0)), mode="constant", constant_values=0)
            voxels.append(v)
        else:
            v = np.asarray(v[:1000])
            voxels.append(v)

    return np.asarray(voxels)
    

def filter_tracks(curr_id, sync_data, extractor):
    imgs = [] # no necessary feature, only for filtering
    voxs = []
    positions = []
    poses = []

    track_dict = {}

    for index, row in sync_data.iterrows():
        img = row["img_msgs"].astype(np.uint8)
        voxel_grid = row["vox_msgs"]
        for object in row["detections"]:
            if type(object[0])== int:
                if object[0] == curr_id:
                    x, y, w, h = object[2]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cropped_img = img.copy()[y:h, x:w]
                    cropped_img = np.asarray(cropped_img.copy(),dtype=np.uint8)
                    cropped_img = pad_img(cropped_img)
                    if len(imgs)>0:
                        features = extractor([imgs[-1], cropped_img]).numpy()
                        m = mse(features[0], features[1])
                        cos = 1 - spatial.distance.cosine(features[0], features[1])
                    
                        if m < 0.6 and cos > 0.8:
                            imgs.append(cropped_img)
                            voxs.append(voxel_grid)
                            positions.append(object[1])
                            poses.append(object[3])
                    else:
                        imgs.append(pad_img(cropped_img))

    track_dict["voxelgrids"] = np.asarray(pad_voxelgrids(voxs))
    track_dict["positions"] = np.asarray(positions)
    track_dict["poses"] = np.asarray(poses)



    if len(imgs) > 10:
        path = "animations/animations_" + str(curr_id) + ".gif"
        imageio.mimsave(path, imgs, duration=len(imgs))
    
    return track_dict


def get_single_tracks(sync_data):
    detections = sync_data.iloc[:]["detections"]
    all_ids = [x[0] for y in detections for x in y]
    occurs_id, occurs_counts = np.unique(np.asarray(all_ids), return_counts=True)
    extractor = torchreid.utils.FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='../osnet_x1_0_imagenet.pth',
        device='cpu'
    )
    tracks = []
    for i in tqdm(range(len(occurs_id))):
        track = filter_tracks(occurs_id[i], sync_data, extractor)
        if len(track["voxelgrids"]) > 10:
            tracks.append(track)
    
    print("tracks: ", len(tracks))
    return tracks




if __name__ == "__main__":

    # load pickle
    sync_data = pd.read_pickle("synced_data/data_2021-04-10-11-28-10-009.pkl")
    tracks = get_single_tracks(sync_data)
    
    print(tracks)

    track_df = pd.DataFrame.from_dict(tracks)
    track_df.to_pickle('full_tracks.pkl')

