import argparse
import json
import numpy as np
import torch
import math
from torch.multiprocessing import set_start_method
import os
from PIL import Image
import json
from tqdm import tqdm
import sys
sys.path.append("./sam2")
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

from sam2.build_sam import build_sam2_video_predictor, build_sam2

OUTPUT = "path/to/output"
OVIS_CHECKPOINT = "path/to/AIDC-AI/Ovis2.5-9B"

def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    all_results = json.load(open(args.all_results_path)) # list

    # create subprocess
    thread_num = args.num_gpus

    processes = []
    lock = mp.Lock()

    video_num = len(all_results)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = all_results[i * per_thread_video_num:]
        else:
            sub_video_list = all_results[i * per_thread_video_num: (i + 1) * per_thread_video_num]

        p = mp.Process(target=sub_processor, args=(lock, i, args, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def save_results(all_boxes, all_points, save_path, frames, img_folder, video_name, save_path_prefix, descriptions, split_frames_idx, exp_id, key_frame_idx, video_predictor):
    if not all_boxes:
        os.makedirs(save_path, exist_ok=True)
        for frame_name in frames:
            img_path = os.path.join(img_folder, video_name, frame_name + ".jpg")
            image = Image.open(img_path).convert("L")
            w, h = image.size
            zero_mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
            save_file = os.path.join(save_path, frame_name + ".png")
            zero_mask.save(save_file)
    else:
        frame_list = [os.path.join(img_folder, video_name, frame_name + ".jpg") for frame_name in frames]

        torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        inference_state = video_predictor.init_state(frame_list=frame_list, offload_video_to_cpu=True)
        selected_frame_ids = [key_frame_idx for _ in range(len(all_boxes))]
        for object_id, (point, box) in enumerate(zip(all_points, all_boxes)):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=selected_frame_ids[object_id],
                obj_id=object_id,
                points=np.array([point], dtype=np.float32),
                labels=np.array([1], np.int32),
                box=box,
            )
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[o] > 0.0).cpu().numpy()
                for o, out_obj_id in enumerate(out_obj_ids)
            }
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[o] > 0.0).cpu().numpy()
                for o, out_obj_id in enumerate(out_obj_ids)
            }

        # save binary image
        # save_path = os.path.join(save_path_prefix, video_name, exp_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for j, segment in video_segments.items():
            mask = np.concatenate(list(segment.values()), axis=0)
            mask = mask.astype(np.float32)
            mask = mask.sum(0)
            mask[mask > 1] = 1
            mask = Image.fromarray(mask * 255).convert('L')
            frame_name = frames[j]
            save_file = os.path.join(save_path, frame_name + ".png")
            mask.save(save_file)

def sub_processor(lock, pid, args, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    for stored_data in video_list:
        save_results(
            all_boxes=stored_data['all_boxes'],
            all_points=stored_data['all_points'],
            save_path=stored_data['save_path'],
            frames=stored_data['frames'],
            img_folder=stored_data['img_folder'],
            video_name=stored_data['video_name'],
            save_path_prefix=stored_data['save_path_prefix'],
            descriptions=stored_data['descriptions'],
            split_frames_idx=stored_data['split_frames_idx'],
            exp_id=stored_data['exp_id'],
            key_frame_idx=stored_data['key_frame_idx'],
            video_predictor=video_predictor
        )

        with lock:
            progress.update(1)
    with lock:
        progress.close()


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_results_path', default='xx', help='path to all_results_path')
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    args = parser.parse_args()

    main(args)