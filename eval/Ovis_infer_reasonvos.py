import argparse
import copy
import glob
import json
import random
import time
import traceback
import cv2
from transformers import AutoModelForCausalLM
import numpy as np
import torch
import re
from torch.multiprocessing import set_start_method
import misc as utils
import torchvision.transforms as T
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from PIL import Image, ImageDraw
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import matplotlib
from typing import List, Dict, Any, Optional
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

from utils.concat_frames_reasonvos import select_top_images, stitch_images, simple_concat_frames
from utils.preprocess_reasonvos import ImageScorer

# colormap
color_list = utils.colormap()
color_list = color_list.astype('uint8').tolist()

color = "#E63946"
color = (np.array(matplotlib.colors.hex2color(color)) * 255).astype('uint8')
edge_width = 15
ratio = 0.99
kernel = np.ones((edge_width, edge_width), np.uint8)

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

OUTPUT = "path/to/output"
OVIS_CHECKPOINT = "path/to/AIDC-AI/Ovis2.5-9B"

def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    seed = 42 + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    save_dir = OUTPUT
    output_dir = os.path.join(save_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)

    # load data
    root = 'path/to/ReasonVOS'
    img_folder = os.path.join(root, 'JPEGImages')
    meta_file = os.path.join(root, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]

    video_list = list(data.keys())

    random.shuffle(video_list)

    all_query_qa = json.load(open('output_query_qa/reasonvos_query_qa.json', 'r'))
    all_CLIP_query_scores = json.load(open('output_concat/reasonvos/CLIP/video_scores_reasonvos.json', 'r'))

    # create subprocess
    thread_num = args.num_gpus
    result_dict = mp.Manager().dict()

    processes = []
    lock = mp.Lock()

    video_num = len(video_list)
    # per_thread_video_num = math.ceil(float(video_num) / float(thread_num))
    per_thread_video_num = video_num // thread_num

    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]

        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   output_dir,
                                                   img_folder, sub_video_list, result_dict, all_query_qa, all_CLIP_query_scores))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    result_dict = dict(result_dict)
    all_results = []
    for pid, cur_results in result_dict.items():
        all_results.extend(cur_results)
    json.dump(all_results, open(os.path.join(output_dir, 'all_results.json'), 'w'), indent=2)
        

# Process video and extract key frames
def extract_key_frames(video_len, num_frames=10):
    total_frames = video_len

    num_frames = min(num_frames, total_frames)

    # Calculate evenly spaced indices
    step = max(1, (total_frames - 1) // (num_frames - 1))
    frame_indices = [i * step for i in range(num_frames)]

    # Ensure last frame is included
    frame_indices[-1] = min(frame_indices[-1], total_frames - 1)

    return frame_indices

score_frame_prompt = """
# Input:
You are given a query and an image that contains frames 1 to 10 of a video. Each frame is labeled with a digit on the top left corner as its frame_id (e.g., '1', '2', '3') indicating its temporal position. The frame_id is very important for subsequent analysis.
'query': {query}
'attention_information'(optional): {attention_info}

# Step-by-Step Processing
1. Describe the events happening in the frame sequence:
   - These frames together form a continuous video segment. They must be interpreted as temporally ordered video frames, not isolated images.
   - You MUST analyze the frames as a sequence to understand motion, temporal behavior, and target dynamics.

2. Score each frame:
   - First, use the frame sequence to understand the query and identify what target object(s) in the video match the query.
   - Then, for each individual frame, evaluate the visual quality and visibility of that target object. Important: You are scoring the target object's appearance in each frame, NOT how well the frame represents the overall event.
   - Consider these factors: Is the target clearly visible or blurred? Is it occluded? Is it large enough and prominently featured? Frames where the target is absent should receive low scores.
   - Rate each frame on a scale of 1 to 10. Higher scores indicate frames where the target object appears most clearly, with minimal blur, no occlusion, and prominent visibility - making it ideal for **detection and segmentation** tasks.

3. Ensure score differentiation:
   - Compare frames against each other. Frames with the best target visibility should receive significantly higher scores (9-10).
   - Frames with poor target visibility or no targets should receive lower scores (1-4).
   - Distribute scores across the full 1-10 range to reflect quality differences between frames.

4. Use the 'attention_information':
   The 'attention_information' (if available) contains guiding information to assist with scoring, primarily indicating what characteristics make a frame suitable or unsuitable as a key frame. You can use this information to help with your scoring. However, you should ignore any specific frame numbers or positional references (like "frame X" or "the X-th frame") mentioned in these prompts, as such information is outdated.

# Output Json Format:
You MUST output ONLY a valid JSON object with the exact structure:
{{"scores": [number1, number2, ...]}}
"""
def score_frame_with_vlm(frames, query, attention_info, vl_model, pid):
    question = score_frame_prompt.format(query=query, attention_info=attention_info)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for key frames analysis."
        },
        {
            "role": "user", "content": [{"type": "image", "image": frame} for frame in frames]
                                        + [{"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=True,
            enable_thinking_budget=True,
            max_new_tokens=4096,
            thinking_budget=3584,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    try:
        # Find the JSON part (after any thinking steps)
        response = response.strip()
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        # Parse JSON to extract target_count and descriptions
        output_json = json.loads(response)
        scores = output_json.get('scores', [])

    except Exception:
        # Fallback if JSON parsing fails
        scores = []

    return scores

generate_descriptions_prompt = """
# Task:
Analyze a spliced image to track objects temporally and generate an unique, key-frame-only visual description for target best matching a flexible query.

# Input:
- A spliced Image: containing {num_frame} frames. The keyframe is the largest, while the remaining frames are smaller. Each frame has a number in the upper left corner, starting from 1, indicating its frame_id in the frame sequence, which is **very important** for subsequent analysis.
- Keyframe ID (corresponds to the number in the upper left corner of a certain frame): {key_frame_id}
- Query: {query}
- History: {history} (Optional. Previous description and evaluator feedback; use to refine by adopting successful elements and avoiding critiqued flaws)

# Step-by-step processing:
## Step 1: Global trajectory analysis
First, examine ALL frames to understand the full context and motion patterns. Based on the query, identify potential candidate objects and track each across the sequence. Then, determine which candidate best matches the query and note its complete trajectory.

## Step 2: Key frame description
Now focus on the specified key frame (ID: {key_frame_id}). For the target identified in Step 1, provide a unique and concise description as it appears in this frame. The description must be sufficient to uniquely identify this target within the key frame.
Focus on **static features** (category, color, appearance, shape, spatial location) visible in the key frame.
The description should be a **single, complete sentence** under 30 words. Do not mention frame numbers or phrases like 'in the key frame'.

## Step 3: Refinement
Based on **history feedback**, refine the description for uniqueness and accuracy if history is provided.

# Output (JSON):
{{"description": "Unique description for the target, limit your response to 30 words."}}
"""
def generate_descriptions_with_vlm(frame, query, key_frame_id, num_frame, vl_model, pid, history=None):
    question = generate_descriptions_prompt.format(query=query, key_frame_id=key_frame_id, history=history, num_frame=num_frame)
    # print("🚀🚀🚀question:  ", question)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for video understanding and descriptions generation."
        },
        {
            "role": "user", "content": [{"type": "image", "image": frame},
                                        {"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=True,
            enable_thinking_budget=True,
            max_new_tokens=4096,
            thinking_budget=3584,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Extract JSON from the response (assuming the model outputs valid JSON)
    try:
        # Find the JSON part (after any thinking steps)
        response = response.strip()
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        # Parse JSON to extract target_count and descriptions
        output_json = json.loads(response)
        description = output_json.get("description", '')

        # Ensure descriptions is a list of unique strings
        descriptions = [description]
    except Exception:
        # Fallback if JSON parsing fails
        descriptions = []

    descriptions = [description.replace(' - ', '-') for description in descriptions]
    return descriptions

generate_descriptions_prompt_origin = """
# Task:
Analyze spliced video frames to track objects temporally and generate an unique, key-frame-only visual description for target best matching a flexible query.

# Input:
- A spliced Image: containing {num_frame} frames. The keyframe is the largest, while the remaining frames are smaller. Each frame has a number in the upper left corner, starting from 1, indicating its frame_id in the frame sequence, which is **very important** for subsequent analysis.
- Keyframe ID (corresponds to the number in the upper left corner of a certain frame): {key_frame_id}
- Query: {query}

# Step-by-step processing:
## Step 1: Global trajectory analysis
First, examine ALL frames to understand the full context and motion patterns. Based on the query, identify potential candidate objects and track each across the sequence. Then, determine which candidate best matches the query and note its complete trajectory.

## Step 2: Key frame description
Now focus on the specified key frame (ID: {key_frame_id}). For the target identified in Step 1, provide a unique and concise description as it appears in this frame. The description must be sufficient to uniquely identify this target within the key frame.
Focus on **static features** (category, color, appearance, shape, spatial location) visible in the key frame.
The description should be a **single, complete sentence** under 30 words. Do not mention frame numbers or phrases like 'in the key frame'.

# Output (JSON):
{{"description": "Unique description for the target, limit your response to 30 words."}}

# Previous results and feedback
The following is a series of reflection steps. Please synthesize all the information to give the final answer.
"""
generate_descriptions_prompt_cot = """
## [Refinement Step] 
In this round, the following shows the generated description and the feedback about the connection and difference between the description and the query '{query}':
Description and feedback: '{update_description_cot}'
"""
def generate_descriptions_with_vlm_cot(frame, query, key_frame_id, update_description_cots, num_frame, vl_model, pid):
    question_origin = generate_descriptions_prompt_origin.format(query=query, key_frame_id=key_frame_id, num_frame=num_frame)
    # print("🚀🚀🚀question:  ", question)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for video understanding and descriptions generation."
        },
        {
            "role": "user", "content": [{"type": "image", "image": frame},
                                        {"type": "text", "text": question_origin}]
        }
    ]

    for update_description_cot in update_description_cots:
        cur_question_cot = generate_descriptions_prompt_cot.format(query=query, update_description_cot=update_description_cot)
        cur_message = [
            {"type": "text", "text": cur_question_cot}
        ]
        messages[-1]['content'].extend(cur_message)

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=True,
            enable_thinking_budget=True,
            max_new_tokens=4096,
            thinking_budget=3584,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Extract JSON from the response (assuming the model outputs valid JSON)
    try:
        # Find the JSON part (after any thinking steps)
        response = response.strip()
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        # Parse JSON to extract target_count and descriptions
        output_json = json.loads(response)
        description = output_json.get("description", '')

        # Ensure descriptions is a list of unique strings
        descriptions = [description]
    except Exception:
        # Fallback if JSON parsing fails
        descriptions = []

    descriptions = [description.replace('-', '') for description in descriptions]
    return descriptions

grounding_prompt = """
# Task
Identify the unique target object in the image that corresponds to the description and provide the bounding box coordinates for that object.

# Input
description: {description}
An original query: {query}

# Instruction
1. Find the <ref>{description}</ref> in the image. Compare the difference between objects and find the **most closely matched one**.
2. The description corresponds to the original query "{query}" in this image, so you can also obtain information about the target from the query if necessary.
3. If you find upon careful consideration that there is no corresponding object in the image, return empty.

Please give the coordinates of the bounding box.
"""
def grounding_with_vlm(frame, query, description, vl_model):
    question = grounding_prompt.format(query=query, description=description)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful image grounding assistant"
        },
        {
            "role": "user", "content": [{"type": "image", "image": frame},
                                        {"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
        max_pixels=896*896
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=False,
            enable_thinking_budget=False,
            max_new_tokens=2048,
            thinking_budget=1024,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    response = response.strip().split('</think>')[-1]
    response = response.replace('<bbox>', '<box>').replace('</bbox>', '</box>')

    box_pattern = r'<box>\s*[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\},\s]*\s*[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}]*\s*</box>'
    box_matches = re.findall(box_pattern, response)

    coord_patterns = [
        # (x1,y1),(x2,y2) or [x1,y1],[x2,y2]
        r'[\(\[\{]\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}][,\s]*[\(\[\{]\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}]',
        # x1,y1,x2,y2
        r'(\d+\.?\d*)\s*[,\s]+\s*(\d+\.?\d*)\s*[,\s]+\s*(\d+\.?\d*)\s*[,\s]+\s*(\d+\.?\d*)',
        # x1 y1 x2 y2
        r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)',
        # four consecutive numbers
        r'\b(\d{2,4})\b\s+\b(\d{2,4})\b\s+\b(\d{2,4})\b\s+\b(\d{2,4})\b'
    ]

    boxes = []

    for match in box_matches:
        try:
            x1, y1, x2, y2 = [float(coord) for coord in match]
            boxes.append(normalize_coordinates(x1, y1, x2, y2))
        except ValueError:
            continue

    for pattern in coord_patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                if len(match) == 4:
                    x1, y1, x2, y2 = [float(coord) for coord in match]
                    boxes.append(normalize_coordinates(x1, y1, x2, y2))
            except ValueError:
                continue

    boxes = remove_duplicate_boxes(boxes)
    if boxes:
        bbox = boxes[-1]
    else:
        bbox = None
    return bbox

def visualize_detection(image, mask=None, box=None):
    if isinstance(image, Image.Image):
        res_img = np.array(image.convert("RGB"))
    else:
        res_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if mask is not None:
        mask_bool = mask.astype(bool)
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        edge_mask = mask_dilated & ~mask_bool
        
        res_img[mask_bool] = (res_img[mask_bool] * ratio + color * (1 - ratio)).astype(np.uint8)
        res_img[edge_mask] = color

    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(res_img, (x1, y1), (x2, y2), color.tolist(), 2)

    return Image.fromarray(res_img)

refine_grounding_prompt = """
You are given an image containing one target outlined by a red bounding box.
You are also given a query '{query}' and a description '{description}' about the image.
Now you need to analyze the image in conjunction with the description, specifically determining whether the target outlined by the red bounding box matches the description.
Based on your analysis, provide the coordinates of the **most critical** point for that target.
Ensure that by combining the point coordinates and the current bounding box, you can accurately pinpoint the target that best matches the description.
The description is the result of the analysis for the query in this image, so you may also appropriately consider information from the query to help you better understand and process the task.
Find the <ref>{description}</ref> in the image. Please provide the point coordinates.
"""
def refine_grounding_with_vlm(frame, query, description, vl_model):
    question = refine_grounding_prompt.format(query=query, description=description)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful image grounding assistant"
        },
        {
            "role": "user", "content": [{"type": "image", "image": frame},
                                        {"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
        max_pixels=896*896
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=False,
            enable_thinking_budget=False,
            max_new_tokens=2048,
            thinking_budget=1024,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    response = response.strip().split('</think>')[-1]
    response = response.replace('<points>', '<point>').replace('</points>', '</point>')
    # response = response.replace('<bbox>', '<box>').replace('</bbox>', '</box>')

    point_pattern = r'<point>\s*[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}]*\s*</point>'
    point_matches = re.findall(point_pattern, response)
    points = []
    for match in point_matches:
        try:
            x, y = [float(coord) for coord in match]
            points.append(normalize_point_coordinates(x, y))
        except ValueError:
            continue
    if points:
        point = points[-1]
    else:
        point = None

    # box_pattern = r'<box>\s*[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\},\s]*\s*[\(\[\{]*\s*([\d\.]+)\s*[,\s]+\s*([\d\.]+)\s*[\)\]\}]*\s*</box>'
    # box_matches = re.findall(box_pattern, response)

    # boxes = []
    # for match in box_matches:
    #     try:
    #         x1, y1, x2, y2 = [float(coord) for coord in match]
    #         boxes.append(normalize_coordinates(x1, y1, x2, y2))
    #     except ValueError:
    #         continue

    # boxes = remove_duplicate_boxes(boxes)
    # if boxes:
    #     bbox = boxes[-1]
    # else:
    #     bbox = None
    return point

def normalize_point_coordinates(x, y):
    if x > 1.0:
        x = x / 1000
    if y > 1.0:
        y = y / 1000

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    return [x, y]

def normalize_coordinates(x1, y1, x2, y2):
    if x1 > 1.0:
        x1 = x1 / 1000
    if x2 > 1.0:
        x2 = x2 / 1000
    if y1 > 1.0:
        y1 = y1 / 1000
    if y2 > 1.0:
        y2 = y2 / 1000

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    return [x1, y1, x2, y2]

def remove_duplicate_boxes(boxes):
    unique_boxes = []
    seen = set()

    for box in boxes:
        rounded_box = tuple(round(coord, 3) for coord in box)
        if rounded_box not in seen:
            seen.add(rounded_box)
            unique_boxes.append(box)

    return unique_boxes

image_qa_prompt = """
# Task Description
You are given a query and an image that contains frames 1 to 5 of a video. Each frame is labeled with a digit **on the top left corner** as its frame_id (e.g., '1', '2', '3') indicating its temporal position. The frame_id is **very important** for subsequent analysis.
- Keyframe ID (corresponds to the number in the upper left corner of a certain frame): {key_frame_id}
- Query: {query}
- Questions: {questions}

# Analysis Instructions
1. Analyze the frames as a continuous video sequence to understand motion, temporal behavior, and target dynamics.
2. For each question in the list, select the optimal answer option based on visual evidence from the video content.
3. Provide brief, factual reasoning for each answer choice, focusing specifically on what is observed in the video frames.

# Key Frame Selection Guidance
After answering all questions, provide targeted guidance for keyframe selection that specifically addresses how to best display targets matching the query '{query}'. The guidance should be:
    - **Focus on query-specific targets**: Based on the query "{query}", identify which specific targets, objects, or elements need to be prominently displayed.
    - **Concrete visual criteria**: Provide specific, actionable criteria for selecting frames that best highlight the targets or how to capture the most representative or informative moment for these targets
IMPORTANT: Do NOT mention specific frame numbers, positions, or references to individual frames.

# Output Format Requirements
You MUST output a valid JSON dictionary with exactly two keys:

1. "answers": A list of dictionaries, each containing:
   - "answer": The selected option letter (e.g., "A", "B")
   - "reason": One sentence providing brief visual explanation focused on the target itself. Do not mention questions or options.

2. "guidance": A concise string providing targeted guidance about key frame selection specifically for displaying query-related targets.

# Example Output Format:
{{
    "answers": [
        {{"answer": "A", "reason": "The target is clearly visible in the center frame without obstruction."}},
        {{"answer": "B", "reason": "Motion blur affects target identification in the sequence."}}
    ],
    "guidance": "Select frames where the red car is fully in frame and not severely obscured by other vehicles. Avoid frames where the car is partially behind trucks or in shadowed areas."
}}
"""
def image_qa_with_vlm(frames, query, questions, key_frame_id, vl_model):
    question = image_qa_prompt.format(query=query, questions=questions, key_frame_id=key_frame_id)

    messages = [
        {
            "role": "system",
            "content": "You are an assistant skilled at answering questions about a series of video frames."
        },
        {
            "role": "user", "content": [{"type": "image", "image": frame} for frame in frames]
                                        + [{"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=True,
            enable_thinking_budget=True,
            max_new_tokens=4096,
            thinking_budget=3584,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Extract JSON from the response (assuming the model outputs valid JSON)
    try:
        # Find the JSON part (after any thinking steps)
        response = response.strip()
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        answer_data = extract_qa_answer_output_for_frame(response)
        if answer_data is None:
            answer_data = {"answers": [], "guidance": ""}
    except Exception:
        # Fallback if JSON parsing fails
        answer_data = {"answers": [], "guidance": ""}

    return answer_data

def extract_qa_answer_output_for_frame(response_text: str) -> Optional[Dict[str, Any]]:
    try:
        def normalize_answer_data(parsed_data):
            if isinstance(parsed_data, dict) and 'answers' in parsed_data:
                for item in parsed_data['answers']:
                    if 'answer' in item:
                        if isinstance(item['answer'], str):
                            item['answer'] = [item['answer']]
                        elif not isinstance(item['answer'], list):
                            item['answer'] = []
            return parsed_data

        try:
            parsed_data = json.loads(response_text)
            if isinstance(parsed_data, dict) and 'answers' in parsed_data and 'guidance' in parsed_data:
                answers = parsed_data['answers']
                if (isinstance(answers, list) and
                        all(isinstance(item, dict) and 'answer' in item and 'reason' in item for item in answers)):
                    parsed_data = normalize_answer_data(parsed_data)
                    valid_answers = all(
                        isinstance(item['answer'], list) and
                        isinstance(item['reason'], str)
                        for item in parsed_data['answers']
                    )
                    if valid_answers:
                        return parsed_data
        except json.JSONDecodeError:
            pass

        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)

        for match in json_matches:
            try:
                parsed_data = json.loads(match)
                if isinstance(parsed_data, dict) and 'answers' in parsed_data and 'guidance' in parsed_data:
                    answers = parsed_data['answers']
                    if (isinstance(answers, list) and
                            all(isinstance(item, dict) and 'answer' in item and 'reason' in item for item in answers)):
                        parsed_data = normalize_answer_data(parsed_data)
                        valid_answers = all(
                            isinstance(item['answer'], list) and
                            isinstance(item['reason'], str)
                            for item in parsed_data['answers']
                        )
                        if valid_answers:
                            return parsed_data
            except json.JSONDecodeError:
                continue

        json_object_pattern = r'\{[\s\S]*?\}'
        json_objects = re.findall(json_object_pattern, response_text)

        for obj_text in json_objects:
            try:
                parsed_data = json.loads(obj_text)
                if isinstance(parsed_data, dict) and 'answers' in parsed_data and 'guidance' in parsed_data:
                    answers = parsed_data['answers']
                    if (isinstance(answers, list) and
                            all(isinstance(item, dict) and 'answer' in item and 'reason' in item for item in answers)):
                        parsed_data = normalize_answer_data(parsed_data)
                        valid_answers = all(
                            isinstance(item['answer'], list) and
                            isinstance(item['reason'], str)
                            for item in parsed_data['answers']
                        )
                        if valid_answers:
                            return parsed_data
            except json.JSONDecodeError:
                continue

        return None

    except Exception as e:
        print(f"Error extracting QA answer output: {e}")
        return None

answer_qa_prompt = """
# Task:
Answer questions about the target in the red bounding box. Output JSON with answer and brief visual reason for each question.

# Input:
Image with red bounding box; Questions dict '{questions}'; Additional information about the questions '{addition_info}'.

# Process:
1. Carefully analyze the image and visual content within the red bounding box.
2. For each question, select the optimal options that accurately describe the visual attributes observed within the bounded area. Provide specific and short factual reasons grounded in the actual image content.
4. Base all responses strictly on concrete visual evidence from the target object within the red box. If the question involves comparisons, such as 'bigger' or 'fastest', judge solely based on the current image and **do not refer to information outside the image**.
5. If the additional information is not empty, you may use it to assist in making selections, as the additional information may refer to relevant details or premises of the question.
6. If some question cannot be answered based on this image, select 'A' and provide the reason.

# Output format example: 
{{"answers": [{{"answer": "A", "reason": "1 sentence for brief visual explanation. Focus on the target itself. Do not mention questions or options!"}}, ...]}}
"""
def answer_qa_with_vlm(frame, questions, addition_info, vl_model):
    if not questions:
        answer_data = {"answers": []}
        return answer_data

    question = answer_qa_prompt.format(questions=questions, addition_info=addition_info)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant for answering questions in an image."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame},
                {"type": "text", "text": question}
            ]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True
    )
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda().to(vl_model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = vl_model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=True,
            enable_thinking_budget=True,
            max_new_tokens=4096,
            thinking_budget=3584,
            do_sample=True,
            eos_token_id=vl_model.text_tokenizer.eos_token_id,
            pad_token_id=vl_model.text_tokenizer.pad_token_id
        )

    response = vl_model.text_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Extract JSON from the response
    try:
        # Find the JSON part (after any thinking steps)
        response = response.strip()
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()

        answer_data = extract_qa_answer_output(response)
        if answer_data is None:
            answer_data = {"answers": []}

    except Exception as e:
        print(f"Error in answer_qa_with_vlm: {e}")
        answer_data = {"answers": []}

    return answer_data

def extract_qa_generation_output(data_string: str) -> Optional[List[Dict[str, Any]]]:
    try:
        return json.loads(data_string)
    except:
        try:
            fixed_str = data_string.replace('{{', '{').replace('}}', '}')
            return json.loads(fixed_str)
        except:
            try:
                pattern = r'\{[^{}]*\{[^{}]*\}[^{}]*\}'
                matches = re.findall(pattern, data_string)
                results = []
                for match in matches:
                    cleaned = match.strip()
                    if cleaned.startswith('{{'):
                        cleaned = '{' + cleaned[2:-2] + '}'
                    results.append(json.loads(cleaned))
                return results
            except:
                pattern = r'"question":\s*"([^"]*)".*?"options":\s*\{([^}]*)\}.*?"correct_answer":\s*"([^"]*)"'
                matches = re.findall(pattern, data_string, re.DOTALL)
                results = []
                for match in matches:
                    question, options_text, answer = match
                    options = {}
                    for opt in ['A', 'B', 'C', 'D']:
                        opt_match = re.search(f'"{opt}":\s*"([^"]*)"', options_text)
                        if opt_match:
                            options[opt] = opt_match.group(1)
                    results.append({
                        'question': question,
                        'options': options,
                        'correct_answer': answer
                    })
                return results

def extract_qa_answer_output(response_text: str) -> Optional[Dict[str, Any]]:
    try:
        def normalize_answer_data(parsed_data):
            if isinstance(parsed_data, dict) and 'answers' in parsed_data:
                for item in parsed_data['answers']:
                    if 'answer' in item:
                        if isinstance(item['answer'], str):
                            item['answer'] = [item['answer']]
                        elif not isinstance(item['answer'], list):
                            item['answer'] = []
            return parsed_data

        try:
            parsed_data = json.loads(response_text)
            if isinstance(parsed_data, dict) and 'answers' in parsed_data:
                answers = parsed_data['answers']
                if (isinstance(answers, list) and
                        all(isinstance(item, dict) and 'answer' in item and 'reason' in item for item in answers)):
                    parsed_data = normalize_answer_data(parsed_data)
                    valid_answers = all(
                        isinstance(item['answer'], list) and
                        isinstance(item['reason'], str)
                        for item in parsed_data['answers']
                    )
                    if valid_answers:
                        return parsed_data
        except json.JSONDecodeError:
            pass

        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)

        for match in json_matches:
            try:
                parsed_data = json.loads(match)
                if isinstance(parsed_data, dict) and 'answers' in parsed_data:
                    answers = parsed_data['answers']
                    if (isinstance(answers, list) and
                            all(isinstance(item, dict) and 'answer' in item and 'reason' in item for item in answers)):
                        parsed_data = normalize_answer_data(parsed_data)
                        valid_answers = all(
                            isinstance(item['answer'], list) and
                            isinstance(item['reason'], str)
                            for item in parsed_data['answers']
                        )
                        if valid_answers:
                            return parsed_data
            except json.JSONDecodeError:
                continue

        json_object_pattern = r'\{[\s\S]*?\}'
        json_objects = re.findall(json_object_pattern, response_text)

        for obj_text in json_objects:
            try:
                parsed_data = json.loads(obj_text)
                if isinstance(parsed_data, dict) and 'answers' in parsed_data:
                    answers = parsed_data['answers']
                    if (isinstance(answers, list) and
                            all(isinstance(item, dict) and 'answer' in item and 'reason' in item for item in answers)):
                        parsed_data = normalize_answer_data(parsed_data)
                        valid_answers = all(
                            isinstance(item['answer'], list) and
                            isinstance(item['reason'], str)
                            for item in parsed_data['answers']
                        )
                        if valid_answers:
                            return parsed_data
            except json.JSONDecodeError:
                continue

        return None

    except Exception as e:
        print(f"Error extracting QA answer output: {e}")
        return None

def convert_qa_for_answer_prompt(qa_data: List[Dict[str, Any]]):
    converted_questions = []

    for description_data in qa_data:
        converted_questions.append({
            'question': description_data.get('question', ''),
            'options': description_data.get('options', {}),
        })

    return converted_questions

def visualize_boxes(image, all_boxes, single_image=True):
    pil_img = Image.fromarray(image.copy())

    if single_image:
        draw = ImageDraw.Draw(pil_img)
        for i, box in enumerate(all_boxes):
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)
        return pil_img
    else:
        images_with_boxes = []
        for box in all_boxes:
            img_copy = pil_img.copy()
            draw = ImageDraw.Draw(img_copy)
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)
            images_with_boxes.append(img_copy)
        return images_with_boxes

def visualize_box(image, box, resize_shape=None):
    pil_img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=5)
    if resize_shape:
        pil_img = pil_img.resize(resize_shape, Image.Resampling.LANCZOS)
    return pil_img

def concat_frames(images, output_path):
    h, w, c = images[0].shape
    fig, axs = plt.subplots(len(images), 1, figsize=(5, 5 * h / w * len(images)))
    if len(images) == 1:
        axs = [axs]
    for i, (ax, img) in enumerate(zip(axs, images), start=1):
        ax.imshow(img)
        circ = Circle((50, 50), 35, fill=True, edgecolor='white', linewidth=2)
        ax.text(50, 50, str(i), color='white', fontsize=15, ha='center', va='center')
        ax.add_patch(circ)
        ax.axis('off')
    plt.tight_layout(pad=0)
    plt.subplots_adjust(hspace=0)
    plt.savefig(output_path)
    plt.close()

def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_j - intersection

        iou = intersection / union

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union

def get_key_frame_id_from_path(combined_image_path):
    key_frame_ids = [int(x) for x in combined_image_path.split('/')[-1].split('.')[0].split('_')[2:]]
    key_frame_idx = key_frame_ids[-1]
    key_frame_id = key_frame_ids.index(key_frame_idx)
    if key_frame_id in [1, 3]:
        key_frame_id += 1
    key_frame_id += 1  # start from 1
    return key_frame_id, key_frame_idx

def convert_bbox(bbox, orig_height, orig_width):
    # abs_x1 = int(bbox[0] / input_width * orig_width)
    # abs_y1 = int(bbox[1] / input_height * orig_height)
    # abs_x2 = int(bbox[2] / input_width * orig_width)
    # abs_y2 = int(bbox[3] / input_height * orig_height)

    abs_x1 = int(bbox[0] * orig_width)
    abs_y1 = int(bbox[1] * orig_height)
    abs_x2 = int(bbox[2] * orig_width)
    abs_y2 = int(bbox[3] * orig_height)

    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1
    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

    abs_x1 = max(0, min(orig_width, abs_x1))
    abs_y1 = max(0, min(orig_height, abs_y1))
    abs_x2 = max(0, min(orig_width, abs_x2))
    abs_y2 = max(0, min(orig_height, abs_y2))

    box_original = [abs_x1, abs_y1, abs_x2, abs_y2]
    return box_original

def convert_point(point, orig_height, orig_width):
    abs_x1 = int(point[0] * orig_width)
    abs_y1 = int(point[1] * orig_height)

    abs_x1 = max(0, min(orig_width, abs_x1))
    abs_y1 = max(0, min(orig_height, abs_y1))

    point_original = [abs_x1, abs_y1]
    return point_original

def sub_processor(lock, pid, args, data, save_path_prefix, img_folder, video_list, result_dict, all_query_qa, all_CLIP_query_scores):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)
    device = torch.device(f"cuda:{pid}")

    ovis_model = AutoModelForCausalLM.from_pretrained(
        OVIS_CHECKPOINT,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device)
    scorer = ImageScorer()

    all_results = []

    # 1. For each video
    for video in video_list:
        torch.cuda.empty_cache()
        metas = []  # list[dict], length is number of expressions

        expressions = data[video]["expressions"]
        num_expressions = len(expressions)
        video_len = len(data[video]["frames"])

        split_frames_list = [data[video]["frames"]]

        for split_frames_idx, frames in enumerate(split_frames_list):
            # read all the anno meta for assigned expressions only
            for i in range(num_expressions):
                meta = {}
                meta["exp"] = expressions[i]["exp_text"]
                meta["exp_id"] = str(expressions[i]['exp_id'])
                meta["frames"] = frames
                metas.append(meta)
            meta = metas

            # store images
            video_name = video
            video_len = len(frames)
            os.makedirs(os.path.join(save_path_prefix, video_name), exist_ok=True)
            imgs = []
            src_imgs = []
            bytes_imgs = []
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                src_img = Image.open(img_path).convert('RGB')
                src_imgs.append(src_img)
                bytes_imgs.append(np.asarray(src_img))

            # 2. For each expression
            for i in range(num_expressions):
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]

                save_path = os.path.join(save_path_prefix, video_name, exp_id)
                # if os.path.exists(save_path):
                #     continue

                start_frame_idx = 1
                end_frame_idx = video_len
                need_update_frame = True
                last_key_frame_idx = 0
                last_boxes = []
                last_points = []
                all_boxes = []
                all_descriptions = []
                attention_info = None
                try:
                    for update_frame_round in range(args.max_update_frame_num):
                        if not need_update_frame:
                            break

                        all_boxes.clear()
                        all_descriptions.clear()

                        CLIP_query_scores = all_CLIP_query_scores[video]['CLIP'][exp_id]
                        CLIP_scores = {}
                        for frame in frames:
                            CLIP_scores[frame + '.jpg'] = CLIP_query_scores[frame + '.jpg']
                        if all_descriptions:
                            CLIP_descriptions_scores = {}
                            for frame_name in CLIP_scores:
                                frame_path = os.path.join(img_folder, video_name, frame_name)
                                CLIP_descriptions_score = 0
                                for description in all_descriptions:
                                    cur_score = scorer.calculate_clip_score(frame_path, description)
                                    CLIP_descriptions_score += cur_score
                                CLIP_descriptions_score /= len(all_descriptions)
                                CLIP_descriptions_scores[frame_name] = CLIP_descriptions_score
                            CLIP_descriptions_scores = scorer.normalize_scores(CLIP_descriptions_scores)

                            for frame_name in CLIP_scores:
                                CLIP_scores[frame_name]['weighted_score'] = 0.1 * CLIP_scores[frame_name]['clip'] + 0.2 * CLIP_descriptions_scores[frame_name]
                        else:
                            for frame_name in CLIP_scores:
                                CLIP_scores[frame_name]['weighted_score'] = 0.1 * CLIP_scores[frame_name]['clip']
                        selected_frames = select_top_images(CLIP_scores, 'weighted_score', start_frame_idx, end_frame_idx)
                        selected_frames_ids = [frames.index(selected_frame.replace('.jpg', '')) for selected_frame in selected_frames]
                        selected_frames_PIL = [src_imgs[selected_frame_id] for selected_frame_id in selected_frames_ids]
                        combined_image = simple_concat_frames(selected_frames_PIL)

                        for query_round in range(args.max_query_num):
                            Ovis_scores = score_frame_with_vlm([combined_image], exp, attention_info, ovis_model, pid)
                            if Ovis_scores and len(Ovis_scores) == len(selected_frames):
                                break

                        if len(Ovis_scores) != len(selected_frames):
                            Ovis_scores = [0] * len(selected_frames)

                        scored_items = []
                        for idx, selected_frame in enumerate(selected_frames):
                            cur_score = 0.0 * CLIP_scores[selected_frame]['weighted_score'] + 0.7 * Ovis_scores[idx]
                            scored_items.append({
                                'original_idx': idx,
                                'score': cur_score
                            })

                        top_5_by_score = sorted(scored_items, key=lambda x: x['score'], reverse=True)[:5]
                        top_5_in_order = sorted(top_5_by_score, key=lambda x: x['original_idx'])

                        keep_indices = [item['original_idx'] for item in top_5_in_order]

                        selected_frames = [selected_frames[i] for i in keep_indices]
                        selected_frames_ids = [selected_frames_ids[i] for i in keep_indices]
                        selected_frames_PIL = [selected_frames_PIL[i] for i in keep_indices]
                        combined_image_frame_qa = simple_concat_frames(selected_frames_PIL)

                        max_score = -1.0
                        key_frame_id = 0
                        for idx, item in enumerate(top_5_in_order):
                            if item['score'] > max_score:
                                max_score = item['score']
                                key_frame_id = idx

                        original_key_frame_id = key_frame_id
                        key_frame_idx = selected_frames_ids[key_frame_id]

                        if key_frame_id in [0, 2, 4]:
                            selected_frame_list = selected_frames_PIL.copy()
                            combined_image = stitch_images(selected_frame_list, key_frame_id)
                        elif key_frame_id == 1:
                            selected_frame_list = selected_frames_PIL.copy()
                            left_pad_idx = (selected_frames_ids[0] + selected_frames_ids[1] + 1) // 2
                            right_pad_idx = (selected_frames_ids[1] + selected_frames_ids[2] + 1) // 2
                            left_pad_frame = Image.open(
                                os.path.join(img_folder, video_name, frames[left_pad_idx] + '.jpg')).convert('RGB')
                            right_pad_frame = Image.open(
                                os.path.join(img_folder, video_name, frames[right_pad_idx] + '.jpg')).convert('RGB')
                            selected_frame_list.insert(1, left_pad_frame)
                            selected_frame_list.insert(3, right_pad_frame)
                            key_frame_id += 1
                            combined_image = stitch_images(selected_frame_list, key_frame_id)
                        else:
                            selected_frame_list = selected_frames_PIL.copy()
                            left_pad_idx = (selected_frames_ids[2] + selected_frames_ids[3] + 1) // 2
                            right_pad_idx = (selected_frames_ids[3] + selected_frames_ids[4] + 1) // 2
                            left_pad_frame = Image.open(
                                os.path.join(img_folder, video_name, frames[left_pad_idx] + '.jpg')).convert('RGB')
                            right_pad_frame = Image.open(
                                os.path.join(img_folder, video_name, frames[right_pad_idx] + '.jpg')).convert('RGB')
                            selected_frame_list.insert(3, left_pad_frame)
                            selected_frame_list.insert(5, right_pad_frame)
                            key_frame_id += 1
                            combined_image = stitch_images(selected_frame_list, key_frame_id)
                        
                        original_key_frame = src_imgs[key_frame_idx]
                        orig_width, orig_height = original_key_frame.size

                        for query_round_description in range(args.max_query_num):
                            descriptions = generate_descriptions_with_vlm(combined_image, exp, key_frame_id + 1, len(selected_frame_list), ovis_model, pid)
                            if descriptions:
                                break
                        
                        all_descriptions = []
                        grounding_bboxes = []
                        grounding_points = []
                        if descriptions:
                            for description in descriptions:
                                for query_round_grounding in range(args.max_query_num):
                                    grounding_bbox = grounding_with_vlm(original_key_frame, exp, description, ovis_model)
                                    
                                    if not grounding_bbox:
                                        continue

                                    converted_box = convert_bbox(grounding_bbox, orig_height, orig_width)
                                    labeled_key_frame = visualize_box(bytes_imgs[key_frame_idx], converted_box)
                                    grounding_point = refine_grounding_with_vlm(labeled_key_frame, exp, description, ovis_model)

                                    if grounding_point and grounding_bbox:
                                        break

                                if grounding_point and grounding_bbox:
                                    grounding_points.append(grounding_point)
                                    grounding_bboxes.append(grounding_bbox)
                                    all_descriptions.append(description)

                        all_boxes = [convert_bbox(bbox, orig_height, orig_width) for bbox in grounding_bboxes]
                        all_points = [convert_point(point, orig_height, orig_width) for point in grounding_points]
                        if grounding_bboxes and grounding_points:
                            last_key_frame_idx = key_frame_idx
                            last_boxes = [convert_bbox(bbox, orig_height, orig_width) for bbox in grounding_bboxes]
                            last_points = [convert_point(point, orig_height, orig_width) for point in grounding_points]

                        if update_frame_round + 1 < args.max_update_frame_num:
                            frame_query_qa = [
                                {
                                    "question": f"Does the frame sequence contain targets that match query '{exp}' but are absent in the keyframe?",
                                    "options": {
                                        "A": "Yes",
                                        "B": "No"
                                    },
                                    "correct_answer": "B"
                                }
                            ]
                            if all_descriptions:
                                frame_query_qa.append(
                                    {
                                        "question": f"Does any other sampled frame present the target corresponding to the description '{all_descriptions[0]}' more clearly than the current keyframe?",
                                        "options": {
                                            "A": "Yes",
                                            "B": "No"
                                        },
                                        "correct_answer": "B"
                                    }
                                )
                                frame_query_qa.append(
                                    {
                                        "question": f"Does the target corresponding to the description '{all_descriptions[0]}' suffers from **severe** occlusion, blurring, or overlap with other objects in the keyframe?",
                                        "options": {
                                            "A": "Yes",
                                            "B": "No"
                                        },
                                        "correct_answer": "B"
                                    }
                                )
                            # Image QA
                            # combined_image_for_check_frame = stitch_images(selected_frame_list, key_frame_id)
                            questions_frame = convert_qa_for_answer_prompt(frame_query_qa)
                            for query_round in range(args.max_query_num):
                                results_for_frame_qa = image_qa_with_vlm([combined_image_frame_qa], exp, questions_frame, original_key_frame_id + 1, ovis_model)
                                if results_for_frame_qa:
                                    break
                            answers_for_frame_qa = results_for_frame_qa['answers']
                            error_fg = 0
                            for qa_data, answer in zip(frame_query_qa, answers_for_frame_qa):
                                if qa_data.get('correct_answer', 'B') not in answer.get('answer', ['B']):
                                    error_fg += 1

                            if error_fg > 0:
                                with open(os.path.join(save_path_prefix, f'feedback_{pid}_frame.txt'), 'a') as f:
                                    f.write(f'{video_name}_{exp_id}\n')
                                need_update_frame = True
                                attention_info = results_for_frame_qa['guidance']
                                continue
                            else:
                                need_update_frame = False

                    need_update_descriptions = True
                    update_descriptions_info = []
                    update_descriptions_cots = []
                    for update_descriptions_round in range(args.max_update_descriptions_num):
                        if not need_update_descriptions:
                            break

                        all_boxes.clear()
                        all_descriptions.clear()

                        if update_descriptions_round > 0:
                            for query_round_description in range(args.max_query_num):
                                if not update_descriptions_cots:
                                    descriptions = generate_descriptions_with_vlm(combined_image, exp, key_frame_id + 1, len(selected_frame_list), ovis_model, pid)
                                else:
                                    descriptions = generate_descriptions_with_vlm_cot(combined_image, exp, key_frame_id + 1, update_descriptions_cots, len(selected_frame_list), ovis_model, pid)

                        grounding_bboxes = []
                        grounding_points = []
                        if descriptions:
                            for description in descriptions:
                                for query_round_grounding in range(args.max_query_num):
                                    grounding_bbox = grounding_with_vlm(original_key_frame, exp, description, ovis_model)

                                    if not grounding_bbox:
                                        continue

                                    converted_box = convert_bbox(grounding_bbox, orig_height, orig_width)
                                    labeled_key_frame = visualize_box(bytes_imgs[key_frame_idx], converted_box)
                                    grounding_point = refine_grounding_with_vlm(labeled_key_frame, exp, description, ovis_model)

                                    if grounding_point and grounding_bbox:
                                        break

                                if grounding_point and grounding_bbox:
                                    grounding_points.append(grounding_point)
                                    grounding_bboxes.append(grounding_bbox)
                                    all_descriptions.append(description)

                        all_boxes = [convert_bbox(bbox, orig_height, orig_width) for bbox in grounding_bboxes]
                        all_points = [convert_point(point, orig_height, orig_width) for point in grounding_points]
                        if grounding_bboxes and grounding_points:
                            last_key_frame_idx = key_frame_idx
                            last_boxes = [convert_bbox(bbox, orig_height, orig_width) for bbox in grounding_bboxes]
                            last_points = [convert_point(point, orig_height, orig_width) for point in grounding_points]

                        if update_descriptions_round + 1 < args.max_update_descriptions_num and all_descriptions and all_boxes:
                            # QA extraction
                            video_exp = video + '/' + str(exp_id)

                            period_count = exp.count('.') + exp.count('?')
                            if period_count <= 1:
                                addition_info = ''
                            else:
                                periods = [i for i, char in enumerate(exp) if char == '.']
                                if len(periods) >= 2:
                                    addition_info = exp[:periods[-2]].strip()
                                else:
                                    first_period_index = exp.find('.')

                                    if first_period_index != -1:
                                        addition_info = exp[:first_period_index].strip()
                                    else:
                                        addition_info = ''

                            query_qa_static = all_query_qa[video_exp].get('static', [])
                            questions_static = convert_qa_for_answer_prompt(query_qa_static)

                            # query QA(Input: img_with_box, QA) (Output: answer, reason)
                            update_descriptions_info.clear()
                            error_cnt = 0
                            for bbox, description in zip(all_boxes, all_descriptions):
                                img_with_box = visualize_box(bytes_imgs[key_frame_idx], bbox)

                                for query_round in range(args.max_query_num):
                                    answers_static = answer_qa_with_vlm(img_with_box, questions_static, addition_info, ovis_model)['answers']
                                    if answers_static:
                                        break

                                reasons_for_cur_description = []
                                error_fg = False
                                for qa_data, answer in zip(query_qa_static, answers_static):
                                    if qa_data.get('correct_answer', 'A') not in answer.get('answer', ['A']):
                                        error_fg = True
                                    reasons_for_cur_description.append(answer.get('reason', ''))

                                if error_fg:
                                    error_cnt += 1
                                reasons_for_cur_description = ' '.join(reasons_for_cur_description)
                                update_descriptions_info.append({
                                    'description': description,
                                    'feedback': reasons_for_cur_description
                                })

                            if error_cnt > 0:
                                need_update_descriptions = True
                                update_descriptions_cots.append(update_descriptions_info)
                            else:
                                need_update_descriptions = False
                        else:
                            need_update_descriptions = True

                    if not all_boxes:
                        key_frame_idx = last_key_frame_idx
                        all_boxes = last_boxes
                        all_points = last_points
                    all_results.append({
                        'all_boxes': all_boxes,
                        'all_points': all_points,
                        'save_path': save_path,
                        'frames': frames,
                        'img_folder': img_folder,
                        'video_name': video_name,
                        'save_path_prefix': save_path_prefix,
                        'descriptions': descriptions,
                        'split_frames_idx': split_frames_idx,
                        'exp_id': exp_id,
                        'key_frame_idx': key_frame_idx
                    })

                except torch.cuda.OutOfMemoryError as e:
                    with open(os.path.join(OUTPUT, f'oom_{pid}.txt'), 'a') as f:
                        f.write(save_path + '\n')
                        f.write(str(e) + '\n')
                        f.write("Traceback:\n")
                        f.write(traceback.format_exc() + '\n\n')
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    with open(os.path.join(OUTPUT, f'error_{pid}.txt'), 'a') as f:
                        f.write(save_path + '\n')
                        f.write(f"Error type: {type(e).__name__}\n")
                        f.write(f"Error message: {str(e)}\n")
                        f.write("Full traceback:\n\n")
                        f.write(traceback.format_exc() + '\n')
                    continue

        with lock:
            progress.update(1)
    result_dict[str(pid)] = all_results
    json.dump(all_results, open(os.path.join(save_path_prefix, f'all_results_{pid}.json'), 'w'), indent=2)
    with lock:
        progress.close()


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='valid',
                        help='valid or valid_u')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--max_query_num", default=3, type=int)
    parser.add_argument("--max_update_descriptions_num", "-max_desc", default=2, type=int)
    parser.add_argument("--max_update_frame_num", "-max_frame", default=2, type=int)
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    args = parser.parse_args()

    main(args)