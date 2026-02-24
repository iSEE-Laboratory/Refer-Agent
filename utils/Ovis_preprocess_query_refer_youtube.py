import argparse
import re
import sys
from typing import Optional, List, Dict, Any

import json
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
from tqdm import tqdm
from torch.multiprocessing import set_start_method
import misc as utils
import numpy as np
import os
import random
import math
import time
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

OUTPUT = "./output_query_qa"
OVIS_CHECKPOINT = "path/to/AIDC-AI/Ovis2.5-9B"


def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    # fix the seed for reproducibility
    seed = 42 + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    save_dir = OUTPUT
    output_dir = save_dir
    os.makedirs(output_dir, exist_ok=True)

    # load data
    root = 'path/to/ref_youtube-vos'
    img_folder = os.path.join(root, split, 'JPEGImages')
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]

    video_list = list(data.keys())

    random.shuffle(video_list)

    # create subprocess
    thread_num = args.num_gpus
    result_dict = mp.Manager().dict()

    processes = []
    lock = mp.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   output_dir,
                                                   img_folder, sub_video_list, result_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    all_results = {}
    for pid, results in result_dict.items():
        all_results.update(results)

    json.dump(all_results, open(os.path.join(output_dir, "reasonvos_query_qa_error.json"), "w"), indent=4)


convert_query_prompt = """
# Subtask1
Given a sentence '{query}'.
If it is an interrogative sentence, convert it into the format of a declarative sentence while minimizing structural changes to the original sentence. Preserve the positions of subject, predicate, object, and adverbial as much as possible while achieving the conversion goal. If the subject cannot be clearly identified, use "target" or "object" as a substitute. Otherwise, use the subject from the query directly. If the query refers to multiple types of definite targets, treat all categories as a single whole.
If the input is a declarative sentence, output it unchanged.
The converted sentence should be a descriptive statement without phrases like "is the one in question" or "is the one in query".

# Subtask2
Classify the converted sentence into one of the following types:
[1] Sentences with clear temporal constraints, especially those describing positions that may change, or actions that cannot be determined from a single image, and those with explicit beginning or end markers (e.g., 'initially', 'at the beginning', 'first', 'finally', 'at the end', 'last'). For such sentences, return type 1.
[2] Sentences where the subject consists of multiple distinct targets, such as a cat and a dog, or a person in white clothes and a person in red clothes, but excluding cases like people in red clothes as they can be categorized as one group. For such sentences, return type 2.
[3] For all other sentences, return type 3.

# Output Json Format
'result': the converted result
'type': the classified type (1, 2, or 3)
"""
def convert_query_with_vlm(query, vl_model):
    question = convert_query_prompt.format(query=query)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user", "content": [{"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True,
        max_pixels=896 * 896
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
            max_new_tokens=2048,
            thinking_budget=1536,
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

        # Parse JSON to extraction
        output_json = json.loads(response)
        result = output_json.get("result", '')
        type = int(output_json.get("type", -1))


    except Exception:
        # Fallback if JSON parsing fails
        result = ''
        type = -1

    return result, type

generate_qa_prompt = """
# Task
Generate 0-3 static question answer sets based on the description provided. Output in JSON format.

# Input:
description: {description}

# Instruction:
1. Identify the main subject from the description. This subject will be referred to as "the target" in all generated questions.
2. All questions must be in "Is" format with exactly two options: "Yes" and "No".
3. Question types to generate are **limited** to the following categories **only if explicitly mentioned in description**:
   - Identity question: "Is the target a [identified subject]?"
   - Appearance question about color, shape, or size
   - Location question about position if the position is static and doesn't change
4. A particular type of question can only be asked once, and relevant information should be presented within a single question without splitting it.

# Rules:
1. Each question must be in "Is" format with exactly two options: "Yes" and "No". The correct answer is always "Yes".
2. If the subject is plural, change it to singular form.
3. **Strictly analyze the description, only generate questions for aspects that are explicitly mentioned in the description. Do not make up any information not present in the description.**
4. Questions must be directly extracted from the description with minimal modification.
5. Each question category corresponds to **one** question. If certain categories are not mentioned in the description, don't generate such questions.

# Output format: 
[
  {{
    "question": "Is the target ...?",
    "options": {{"A": "Yes", "B": "No"}},
    "correct_answer": "A"
  }},
  ...
]
"""

def generate_qa_with_vlm(description, type, vl_model):
    if type == 3:
        question = generate_qa_prompt.format(description=description)
    else:
        return []

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": question}]
        }
    ]

    input_ids, pixel_values, grid_thws = vl_model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=True,
        max_pixels=896 * 896
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
            max_new_tokens=2048,
            thinking_budget=1536,
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

        qa_data = extract_qa_generation_output(response)
        if qa_data is None:
            qa_data = []

    except Exception as e:
        print(f"Error in generate_qa_with_vlm: {e}")
        qa_data = []

    return qa_data

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

def sub_processor(lock, pid, args, data, save_path_prefix, img_folder, video_list, result_dict):
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

    results = {}

    # 1. For each video
    for video in video_list:
        torch.cuda.empty_cache()
        metas = []  # list[dict], length is number of expressions

        expressions = data[video]["expressions"]

        # read all the anno meta
        for exp_id in expressions:
            meta = {}
            meta["exp"] = expressions[exp_id]["exp"]
            meta["exp_id"] = exp_id
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for exp_id in expressions:
            exp = meta[exp_id]["exp"]
            exp_id = meta[exp_id]["exp_id"]
            video_exp = f'{video}/{exp_id}'

            # exp = 'Which shoe(s) is/are on the right foot of the man?'

            try:
                exp = exp.strip()
                for query_round in range(args.max_query_num):
                    description, type = convert_query_with_vlm(exp, ovis_model)
                    if description and type != -1:
                        break

                if not exp.strip().endswith('?'):
                    description = exp.strip()
                else:
                    description = description.strip()

                if type == 3:
                    for query_round in range(args.max_query_num):
                        qa_complete = generate_qa_with_vlm(description, type, ovis_model)
                        if qa_complete:
                            break
                else:
                    qa_complete = []
                
                results[video_exp] = {
                    'exp': exp,
                    'description': description,
                    'type': type,
                    'qa_complete': qa_complete,
                }

            except Exception as e:
                with open(f'error_{pid}.txt', 'a') as f:
                    f.write(str(e) + '\n')
                    f.write(f'{video}/{exp_id}\n')
                continue

        with lock:
            progress.update(1)

    # json.dump(results, open(os.path.join(save_path_prefix, f'result_{pid}.json'), 'w'), indent=2)

    result_dict[str(pid)] = results
    with lock:
        progress.close()


if __name__ == '__main__':
    set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='valid',
                        help='valid or valid_u')
    parser.add_argument("--refine_round", default=0, type=int)
    parser.add_argument("--max_query_num", default=3, type=int)
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    args = parser.parse_args()

    main(args)