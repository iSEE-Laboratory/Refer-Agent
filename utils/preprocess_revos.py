#!/usr/bin/env python3
import json
import os
import argparse
from typing import Dict
import warnings
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from pyiqa import create_metric
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

warnings.filterwarnings('ignore')


class ImageScorer:
    def __init__(self, clip_model_name="path/to/openai/clip-vit-base-patch32",
                 iqa_metric_name="brisque", iqa_metric_mode="NR"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize CLIP model
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.to(self.device)

        # Initialize IQA model
        self.iqa_model = create_metric(iqa_metric_name, metric_mode=iqa_metric_mode, device=self.device)

    def calculate_clip_score(self, image_path: str, text: str) -> float:
        if not text.strip():
            return 0.0  # Return raw score for empty text

        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            raw_score = outputs.logits_per_image.cpu().numpy()[0][0]

        return float(raw_score)  # Return raw score for later normalization

    def calculate_iqa_score(self, image_path: str) -> float:
        score = self.iqa_model(image_path).cpu().item()
        return float(score)  # Return raw score for later normalization

    def normalize_scores(self, scores_dict: Dict[str, float], invert: bool = False) -> Dict[str, float]:
        """Normalize scores to 1-10 scale using mean and std"""
        if not scores_dict:
            return {}

        scores = list(scores_dict.values())
        mean_score = sum(scores) / len(scores)
        std_score = (sum((x - mean_score) ** 2 for x in scores) / len(scores)) ** 0.5

        # Avoid division by zero
        if std_score == 0:
            return {k: 5.5 for k in scores_dict.keys()}

        normalized = {}
        for key, score in scores_dict.items():
            # Z-score normalization then map to 1-10
            z_score = (score - mean_score) / std_score
            if invert:  # For metrics where lower is better (like BRISQUE)
                z_score = -z_score
            # Map z-score to 1-10 range (assuming z-scores mostly in [-3, 3])
            norm_score = 5.5 + 1.5 * z_score
            normalized[key] = max(1.0, min(10.0, norm_score))

        return normalized


def batch_score_videos(base_dir: str,
                       weights: Dict[str, float] = None,
                       iqa_metric_name: str = "brisque") -> Dict[str, Dict]:
    """Batch score all image frames in multiple video directories"""
    meta = json.load(open('path/to/revos/meta_expressions_valid_.json'))['videos']

    scorer = ImageScorer(iqa_metric_name=iqa_metric_name)
    video_scores = {}

    if weights is None:
        weights = {'clip': 1.0, 'iqa': 0}

    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    for video_dir_name in tqdm(meta.keys()):
        video_path = os.path.join(base_dir, video_dir_name)
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

        image_files = sorted([f for f in os.listdir(video_path)
                              if f.lower().endswith(supported_formats)],
                             key=lambda x: int(''.join(filter(str.isdigit, x))))

        if not image_files:
            print(f"No supported image files found in '{video_dir_name}'.")
            continue

        # print(f"\nProcessing video: {video_dir_name}")

        # Initialize video scores structure
        video_scores[video_dir_name] = {}
        video_scores[video_dir_name]['CLIP'] = {}

        # Calculate raw IQA scores once for all images in this video
        # print("Calculating IQA scores...")
        raw_iqa_scores = {}
        for image_file in (image_files):
            image_path = os.path.join(video_path, image_file)
            raw_iqa_scores[image_file] = scorer.calculate_iqa_score(image_path)

        # Normalize IQA scores for this video
        # For BRISQUE, lower is better, so we use invert=True
        invert_iqa = iqa_metric_name.lower() in ['brisque', 'niqe']  # Add other metrics that need inversion
        normalized_iqa_scores = scorer.normalize_scores(raw_iqa_scores, invert=invert_iqa)
        video_scores[video_dir_name]['IQA'] = normalized_iqa_scores

        # Calculate CLIP scores for each expression
        for exp_id in meta[video_dir_name]['expressions'].keys():
            # print(f"Processing expression {exp_id}...")
            reference_text = meta[video_dir_name]['expressions'][exp_id]['exp']
            raw_clip_scores = {}

            # First pass: calculate all raw CLIP scores for this expression
            for image_file in (image_files):
                image_path = os.path.join(video_path, image_file)
                raw_clip_scores[image_file] = scorer.calculate_clip_score(image_path, reference_text)

            # Normalize CLIP scores for this expression
            normalized_clip_scores = scorer.normalize_scores(raw_clip_scores, invert=False)

            # Calculate weighted scores
            exp_scores = {}
            for image_file in image_files:
                clip_score = normalized_clip_scores[image_file]
                iqa_score = normalized_iqa_scores[image_file]
                weighted_score = weights['clip'] * clip_score + weights['iqa'] * iqa_score

                exp_scores[image_file] = {
                    'clip': round(clip_score, 2),
                    'weighted_score': round(weighted_score, 2)
                }

            video_scores[video_dir_name]['CLIP'][exp_id] = exp_scores

    return video_scores


def main():
    parser = argparse.ArgumentParser(description='Video frame quality scoring tool.')
    parser.add_argument('--input_dir', default='path/to/revos',
                        help='Path to the directory containing video frame folders.')
    parser.add_argument('--weights', '-w', default='1.0,0',
                        help='Weights for (clip, iqa).')
    parser.add_argument('--output', '-o', default='path/to/video_scores_revos.json',
                        help='Output JSON file path.')
    parser.add_argument('--iqa_metric', default='brisque',
                        help='IQA metric name from pyiqa.')

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory not found - {args.input_dir}")
        return

    weight_values = [float(w) for w in args.weights.split(',')]
    if len(weight_values) != 2:
        print("Invalid weight format, using default weights.")
        weights = {'clip': 1.0, 'iqa': 0}
    else:
        weights = {
            'clip': weight_values[0],
            'iqa': weight_values[1]
        }

    print(f"Starting batch scoring of video frames in '{args.input_dir}'...")
    print(f"Using IQA metric: {args.iqa_metric}")

    all_video_scores = batch_score_videos(args.input_dir, weights, args.iqa_metric)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_video_scores, f, ensure_ascii=False, indent=2)

    print(f"\nAll results saved to: {args.output}")


if __name__ == "__main__":
    main()