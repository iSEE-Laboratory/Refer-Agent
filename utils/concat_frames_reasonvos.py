import glob
import os
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm
import shutil
import numpy as np


def select_uniform_images(image_dict, type='clip', start_idx=1, end_idx=-1, num_frame=10):
    all_images = list(image_dict.keys())
    if end_idx == -1:
        end_idx = len(all_images)

    if start_idx < 1 or end_idx > len(all_images) or start_idx > end_idx:
        raise ValueError("Invalid start index or end index")

    interval_images = all_images[start_idx - 1:end_idx]

    total_images = len(interval_images)

    if total_images <= num_frame:
        return sorted(interval_images)

    selected_indices = np.linspace(0, total_images - 1, num_frame, dtype=int).tolist()
    selected_images = [interval_images[i] for i in selected_indices]

    return sorted(selected_images)


def select_top_score_images(image_dict, type='clip', start_idx=1, end_idx=-1, num_frame=10):
    all_images = list(image_dict.keys())
    if end_idx == -1:
        end_idx = len(all_images)

    if start_idx < 1 or end_idx > len(all_images) or start_idx > end_idx:
        raise ValueError("Invalid start index or end index")

    interval_images = all_images[start_idx - 1:end_idx]

    if len(interval_images) <= num_frame:
        return sorted(interval_images)

    top_images = sorted(interval_images,
                        key=lambda img: image_dict[img][type],
                        reverse=True)[:num_frame]

    return sorted(top_images)

def select_top_images(image_dict, type='clip', start_idx=1, end_idx=-1, num_frame=10):
    all_images = list(image_dict.keys())
    if end_idx == -1:
        end_idx = len(all_images)

    if start_idx < 1 or end_idx > len(all_images) or start_idx > end_idx:
        raise ValueError("Invalid start index or end index")

    interval_images = all_images[start_idx - 1:end_idx]

    total_images = len(interval_images)
    subinterval_size = total_images // num_frame
    remainder = total_images % num_frame

    subintervals = []
    start = 0
    for i in range(num_frame):
        size = subinterval_size + (1 if i < remainder else 0)
        end = start + size
        subintervals.append(interval_images[start:end])
        start = end

    selected_images = []
    for subinterval in subintervals:
        if subinterval:
            top_image = max(subinterval, key=lambda img: image_dict[img][type])
            selected_images.append(top_image)

    return selected_images


def stitch_images(ori_image_list, key_frame_index, img_size=None):
    image_list = []
    for ori_image in ori_image_list:
        cur_image = ori_image.copy()
        if img_size:
            cur_image.thumbnail(img_size, Image.Resampling.LANCZOS)
        image_list.append(cur_image)
    num_images = len(image_list)

    w, h = image_list[0].size

    small_w = int(w * 0.495)
    small_h = int(h * 0.495)

    gap_w = int(w * 0.01)
    gap_h = int(h * 0.01)

    def add_number_label(image, number, is_key_frame=False):
        # Create a copy of the image to avoid modifying the original
        img = image.copy()
        draw = ImageDraw.Draw(img)

        # Get image dimensions
        img_width, img_height = img.size

        # Define circle parameters
        circle_radius = min(img_width, img_height) // 25  # 5% of smaller dimension
        circle_radius = int(circle_radius * 1.5) if not is_key_frame else circle_radius
        circle_center = (circle_radius + 10, circle_radius + 10)  # 10px padding from top-left
        circle_bbox = [
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius
        ]

        # Draw opaque circle (red for keyframe, blue for non-keyframe)
        circle_color = (255, 100, 100) if is_key_frame else (0, 150, 255)
        draw.ellipse(circle_bbox, fill=circle_color)

        # Load a default font (or system font if available)
        try:
            font_path = "arial.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font_size = int(circle_radius * (1.2 if is_key_frame else 1.0))
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            font_size = int(circle_radius * (1.2 if is_key_frame else 1.0))
            font = ImageFont.load_default().font_variant(size=font_size)

        # Get text size for centering
        text = str(number)
        text_bbox = draw.textbbox((0, 0), text, font=font, anchor='mm')
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate text position to strictly center in circle
        text_pos = (
            circle_center[0],
            circle_center[1]
        )

        # Draw white text, centered using 'mm' anchor
        draw.text(text_pos, text, fill='white', font=font, anchor='mm')

        return img

    if num_images == 5:
        if key_frame_index == 0:
            canvas_w = w + gap_w + 2 * small_w + gap_w
            canvas_h = max(h, 2 * small_h + gap_h)
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            key_y = (canvas_h - h) // 2
            key_img = add_number_label(image_list[0], 1, True)
            canvas.paste(key_img, (0, key_y))

            right_start_x = w + gap_w
            positions = [
                (right_start_x, 0),
                (right_start_x, small_h + gap_h),
                (right_start_x + small_w + gap_w, 0),
                (right_start_x + small_w + gap_w, small_h + gap_h)
            ]

            for i, pos in enumerate(positions):
                img = image_list[i + 1].resize((small_w, small_h))
                img = add_number_label(img, i + 2)
                canvas.paste(img, pos)

        elif key_frame_index == 2:
            canvas_w = small_w + gap_w + w + gap_w + small_w
            canvas_h = max(h, 2 * small_h + gap_h)
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            img1 = image_list[0].resize((small_w, small_h))
            img1 = add_number_label(img1, 1)
            canvas.paste(img1, (0, 0))

            img2 = image_list[1].resize((small_w, small_h))
            img2 = add_number_label(img2, 2)
            canvas.paste(img2, (0, small_h + gap_h))

            key_y = (canvas_h - h) // 2
            key_img = add_number_label(image_list[2], 3, True)
            canvas.paste(key_img, (small_w + gap_w, key_y))

            right_x = small_w + gap_w + w + gap_w
            img4 = image_list[3].resize((small_w, small_h))
            img4 = add_number_label(img4, 4)
            canvas.paste(img4, (right_x, 0))

            img5 = image_list[4].resize((small_w, small_h))
            img5 = add_number_label(img5, 5)
            canvas.paste(img5, (right_x, small_h + gap_h))

        else:  # key_frame_index == 4
            canvas_w = 2 * small_w + gap_w + gap_w + w
            canvas_h = max(h, 2 * small_h + gap_h)
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            positions = [
                (0, 0),
                (0, small_h + gap_h),
                (small_w + gap_w, 0),
                (small_w + gap_w, small_h + gap_h)
            ]

            for i, pos in enumerate(positions):
                img = image_list[i].resize((small_w, small_h))
                img = add_number_label(img, i + 1)
                canvas.paste(img, pos)

            key_x = 2 * small_w + 2 * gap_w
            key_y = (canvas_h - h) // 2
            key_img = add_number_label(image_list[4], 5, True)
            canvas.paste(key_img, (key_x, key_y))

    elif num_images == 7:
        if key_frame_index == 2:
            canvas_w = small_w + gap_w + w + gap_w + 2 * small_w + gap_w
            canvas_h = max(h, 2 * small_h + gap_h)
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            img1 = image_list[0].resize((small_w, small_h))
            img1 = add_number_label(img1, 1)
            canvas.paste(img1, (0, 0))

            img2 = image_list[1].resize((small_w, small_h))
            img2 = add_number_label(img2, 2)
            canvas.paste(img2, (0, small_h + gap_h))

            key_y = (canvas_h - h) // 2
            key_img = add_number_label(image_list[2], 3, True)
            canvas.paste(key_img, (small_w + gap_w, key_y))

            right_x = small_w + gap_w + w + gap_w
            positions = [
                (right_x, 0),
                (right_x, small_h + gap_h),
                (right_x + small_w + gap_w, 0),
                (right_x + small_w + gap_w, small_h + gap_h)
            ]

            for i, pos in enumerate(positions):
                img = image_list[i + 3].resize((small_w, small_h))
                img = add_number_label(img, i + 4)
                canvas.paste(img, pos)

        else:  # key_frame_index == 4
            canvas_w = 2 * small_w + gap_w + gap_w + w + gap_w + small_w
            canvas_h = max(h, 2 * small_h + gap_h)
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            positions = [
                (0, 0),
                (0, small_h + gap_h),
                (small_w + gap_w, 0),
                (small_w + gap_w, small_h + gap_h)
            ]

            for i, pos in enumerate(positions):
                img = image_list[i].resize((small_w, small_h))
                img = add_number_label(img, i + 1)
                canvas.paste(img, pos)

            key_x = 2 * small_w + 2 * gap_w
            key_y = (canvas_h - h) // 2
            key_img = add_number_label(image_list[4], 5, True)
            canvas.paste(key_img, (key_x, key_y))

            right_x = key_x + w + gap_w
            img6 = image_list[5].resize((small_w, small_h))
            img6 = add_number_label(img6, 6)
            canvas.paste(img6, (right_x, 0))

            img7 = image_list[6].resize((small_w, small_h))
            img7 = add_number_label(img7, 7)
            canvas.paste(img7, (right_x, small_h + gap_h))

    return canvas


def simple_concat_frames(ori_image_list, key_frame_index=0, img_size=None):
    if not ori_image_list:
        return None

    image_list = []
    for ori_image in ori_image_list:
        image = ori_image.copy()
        if img_size:
            image.thumbnail(img_size, Image.Resampling.LANCZOS)
        image_list.append(image)

    w, h = image_list[0].size

    labeled_images = []
    for i, img in enumerate(image_list):
        is_key = False
        labeled_img = add_simple_number_label(img.copy(), i + 1, is_key)
        labeled_images.append(labeled_img)

    if h <= w:
        total_width = w * len(labeled_images)
        total_height = h
        canvas = Image.new('RGB', (total_width, total_height), (240, 240, 240))

        for i, img in enumerate(labeled_images):
            canvas.paste(img, (i * w, 0))
    else:
        total_width = w
        total_height = h * len(labeled_images)
        canvas = Image.new('RGB', (total_width, total_height), (240, 240, 240))

        for i, img in enumerate(labeled_images):
            canvas.paste(img, (0, i * h))

    return canvas


def add_simple_number_label(image, number, is_key_frame=False):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    # Get image dimensions
    img_width, img_height = img.size

    # Define circle parameters
    circle_radius = min(img_width, img_height) // 25
    circle_center = (circle_radius + 10, circle_radius + 10)
    circle_bbox = [
        circle_center[0] - circle_radius,
        circle_center[1] - circle_radius,
        circle_center[0] + circle_radius,
        circle_center[1] + circle_radius
    ]

    # Draw opaque circle (red for keyframe, blue for non-keyframe)
    circle_color = (255, 100, 100) if is_key_frame else (0, 150, 255)
    draw.ellipse(circle_bbox, fill=circle_color)

    # Load a default font
    try:
        font_path = "arial.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_size = int(circle_radius * 1.2)
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Draw text
    text = str(number)
    text_pos = (circle_center[0], circle_center[1])
    draw.text(text_pos, text, fill='white', font=font, anchor='mm')

    return img