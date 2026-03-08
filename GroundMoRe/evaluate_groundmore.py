import os
import json
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017.utils import db_statistics

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def time_str_to_seconds(time_str):
    """Converts a time string to seconds."""
    parts = time_str.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main():
    # initialize the J&F res
    J_list = []
    F_list = []

    causal_J_list = []
    causal_F_list = []
    sequential_J_list = []
    sequential_F_list = []
    counterfactual_J_list = []
    counterfactual_F_list = []
    descriptive_J_list = []
    descriptive_F_list = []

    # load data
    video_root = "path/to/GroundMoRe/annotations"
    meta_file = "path/to/GroundMoRe/test_v2.json"
    annotation_dir = 'path/to/output'

    file = open(os.path.join(annotation_dir, '..', 'result.txt'), 'a')

    with open(meta_file, "r") as f:
        metadata = json.load(f)["videos"]

    video_list = list(metadata.keys())

    # 1. For each video
    for video in tqdm(video_list):
        metas = [] # list[dict], length is number of expressions

        expressions = metadata[video]["questions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)

        clip_start = video[-9:].split("_")[0][:2] + ":" + video[-9:].split("_")[0][2:]
        clip_end = video[-9:].split("_")[1][:2] + ":" + video[-9:].split("_")[1][2:]

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video_id"] = video
            meta["exp"] = expressions[expression_list[i]]["question"]
            meta["ans"] = expressions[expression_list[i]]["answer"]
            meta["obj_id"] = int(expressions[expression_list[i]]["obj_id"])
            meta["q_type"] = expressions[expression_list[i]]["q_type"]
            meta["exp_id"] = expression_list[i]

            start = expressions[expression_list[i]]["action_start"]
            end = expressions[expression_list[i]]["action_end"]
            action_start = (time_str_to_seconds(start) - time_str_to_seconds(clip_start)) * 6 # fps=6
            action_end = (time_str_to_seconds(end) - time_str_to_seconds(clip_start)) * 6 - 1

            meta["action_start"] = action_start
            meta["action_end"] = action_end
            # meta["frame_dir"] = frame_start.zfill(4) + "_" + frame_end.zfill(4)
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_id = meta[i]["video_id"]
            exp_id = meta[i]["exp_id"]
            obj_id = meta[i]["obj_id"]
            q_type = meta[i]["q_type"]

            # action start and end is used to obtain gt masks in temporal dimension
            action_start = meta[i]["action_start"]  
            action_end = meta[i]["action_end"]

            frame_dir = os.path.join(video_root, video_id, "images/")
            if not os.path.exists(frame_dir):
                print("Missing frames: {}.".format(video_id))
                continue
            raw_frames = [x for x in sorted(os.listdir(frame_dir)) if x.endswith('.jpg')]  # all the frames
            # sample_indices = np.linspace(0, len(raw_frames) - 1, num=20, dtype=int)  # uniformly sample 20 frames
            # frames = [raw_frames[i] for i in sample_indices]
            frames = raw_frames
            video_len = len(frames)

            image_np_list = []
            original_size_list = []

            for t in range(video_len):
                frame_id = frames[t]
                image_path = os.path.join(frame_dir, frame_id)

                image_np = cv2.imread(image_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                image_np_list.append(image_np)
                original_size_list.append(image_np.shape[:2])

            origin_h, origin_w = original_size_list[0]
            all_pred_masks = np.zeros((video_len, origin_h, origin_w), dtype=np.uint8)
            for j, frame in enumerate(frames):
                frame_name = frame.replace('jpg', 'png')
                frame_path = os.path.join(annotation_dir, video_id, exp_id, frame_name)
                if os.path.exists(frame_path):
                    all_pred_masks[j] = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                else:
                    all_pred_masks[j] = np.zeros((origin_h, origin_w), dtype=np.uint8)

            # load gt masks
            mask_dir = os.path.join(video_root, video_id, "masks/")
            gt_masks_list = []
            # for index in sample_indices:
            for index in range(video_len):
                if action_start <= index <= action_end:
                    mask_id = "frame_" + str(index).zfill(6) + ".png"
                    mask_path = os.path.join(mask_dir, mask_id)
                    if os.path.exists(mask_path):
                        raw_mask = Image.open(mask_path).convert('P')
                    else:
                        raw_mask = np.zeros((origin_h, origin_w), dtype=np.int32)  # need to add frame index annotation in the meta file
                    raw_mask = np.array(raw_mask)
                    gt_mask = (raw_mask==obj_id).astype(np.float32)
                else:
                    gt_mask = np.zeros((origin_h, origin_w), dtype=np.int32)

                gt_masks_list.append(gt_mask) # list[mask]
            gt_masks = np.stack(gt_masks_list, axis=0)

            # calculate J & F
            j_metric = db_eval_iou(gt_masks, all_pred_masks)
            f_metric = db_eval_boundary(gt_masks, all_pred_masks)
            [JM, JR, JD] = db_statistics(j_metric)
            [FM, FR, FD] = db_statistics(f_metric)

            # print(video_id, JM, FM)
            JF = (JM + FM) / 2
            file.write(f'{video_id}, {exp_id}, {JF}, {JM}, {FM}\n')

            J_list.append(JM)
            F_list.append(FM)

            if q_type == "Causal":
                causal_J_list.append(JM)
                causal_F_list.append(FM)
            elif q_type == "Sequential":
                sequential_J_list.append(JM)
                sequential_F_list.append(FM)
            elif q_type == "Counterfactual":
                counterfactual_J_list.append(JM)
                counterfactual_F_list.append(FM)
            elif q_type == "Descriptive":
                descriptive_J_list.append(JM)
                descriptive_F_list.append(FM)

    final_J = np.mean(J_list)
    final_F = np.mean(F_list)
    final_JF = (final_J + final_F) / 2

    final_causal_J = np.mean(causal_J_list)
    final_causal_F = np.mean(causal_F_list)
    final_sequential_J = np.mean(sequential_J_list)
    final_sequential_F = np.mean(sequential_F_list)
    final_counterfactual_J = np.mean(counterfactual_J_list)
    final_counterfactual_F = np.mean(counterfactual_F_list)
    final_descriptive_J = np.mean(descriptive_J_list)
    final_descriptive_F = np.mean(descriptive_F_list)

    final_causal_JF = (final_causal_J + final_causal_F) / 2
    final_sequential_JF = (final_sequential_J + final_sequential_F) / 2
    final_counterfactual_JF = (final_counterfactual_J + final_counterfactual_F) / 2
    final_descriptive_JF = (final_descriptive_J + final_descriptive_F) / 2

    with open(os.path.join(annotation_dir, 'results_all.txt'), "w") as f:
        f.write(f"Final J (Jaccard Index): {final_J:.4f}\n")
        f.write(f"Final F (F-measure): {final_F:.4f}\n")
        f.write(f"Final JF (Average of J and F): {final_JF:.4f}\n\n")

        f.write(f"Final Causal J: {final_causal_J:.4f}\n")
        f.write(f"Final Causal F: {final_causal_F:.4f}\n")
        f.write(f"Final Causal JF (Average of Causal J and F): {final_causal_JF:.4f}\n\n")

        f.write(f"Final Sequential J: {final_sequential_J:.4f}\n")
        f.write(f"Final Sequential F: {final_sequential_F:.4f}\n")
        f.write(f"Final Sequential JF (Average of Sequential J and F): {final_sequential_JF:.4f}\n\n")

        f.write(f"Final Counterfactual J: {final_counterfactual_J:.4f}\n")
        f.write(f"Final Counterfactual F: {final_counterfactual_F:.4f}\n")
        f.write(f"Final Counterfactual JF (Average of Counterfactual J and F): {final_counterfactual_JF:.4f}\n\n")

        f.write(f"Final Descriptive J: {final_descriptive_J:.4f}\n")
        f.write(f"Final Descriptive F: {final_descriptive_F:.4f}\n")
        f.write(f"Final Descriptive JF (Average of Descriptive J and F): {final_descriptive_JF:.4f}")

if __name__ == "__main__":
    main()
