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
        img = image.copy()
        draw = ImageDraw.Draw(img)

        img_width, img_height = img.size

        circle_radius = min(img_width, img_height) // 25
        circle_radius = int(circle_radius * 1.5) if not is_key_frame else circle_radius
        circle_center = (circle_radius + 10, circle_radius + 10)
        circle_bbox = [
            circle_center[0] - circle_radius,
            circle_center[1] - circle_radius,
            circle_center[0] + circle_radius,
            circle_center[1] + circle_radius
        ]

        circle_color = (255, 100, 100) if is_key_frame else (0, 150, 255)
        draw.ellipse(circle_bbox, fill=circle_color)

        try:
            font_path = "arial.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font_size = int(circle_radius * (1.2 if is_key_frame else 1.0))
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            font_size = int(circle_radius * (1.2 if is_key_frame else 1.0))
            font = ImageFont.load_default().font_variant(size=font_size)

        text = str(number)
        text_bbox = draw.textbbox((0, 0), text, font=font, anchor='mm')

        text_pos = (circle_center[0], circle_center[1])
        draw.text(text_pos, text, fill='white', font=font, anchor='mm')

        return img

    row_small_w = 2 * small_w + gap_w
    canvas_w    = max(w, row_small_w)

    key_x   = (canvas_w - w) // 2
    small_x = (canvas_w - row_small_w) // 2

    def paste_small_row(canvas, imgs, numbers, top_y):
        for col, (img, num) in enumerate(zip(imgs, numbers)):
            resized = img.resize((small_w, small_h))
            labeled = add_number_label(resized, num)
            x = small_x + col * (small_w + gap_w)
            canvas.paste(labeled, (x, top_y))

    if num_images == 5:
        if key_frame_index == 0:
            canvas_h = h + gap_h + 2 * small_h + gap_h
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            key_img = add_number_label(image_list[0], 1, True)
            canvas.paste(key_img, (key_x, 0))

            paste_small_row(canvas, [image_list[1], image_list[2]], [2, 3],
                            h + gap_h)
            paste_small_row(canvas, [image_list[3], image_list[4]], [4, 5],
                            h + gap_h + small_h + gap_h)

        elif key_frame_index == 2:
            canvas_h = small_h + gap_h + h + gap_h + small_h
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            paste_small_row(canvas, [image_list[0], image_list[1]], [1, 2], 0)

            key_img = add_number_label(image_list[2], 3, True)
            canvas.paste(key_img, (key_x, small_h + gap_h))

            paste_small_row(canvas, [image_list[3], image_list[4]], [4, 5],
                            small_h + gap_h + h + gap_h)

        else:  # key_frame_index == 4
            canvas_h = 2 * small_h + gap_h + gap_h + h
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            paste_small_row(canvas, [image_list[0], image_list[1]], [1, 2], 0)
            paste_small_row(canvas, [image_list[2], image_list[3]], [3, 4],
                            small_h + gap_h)

            key_img = add_number_label(image_list[4], 5, True)
            canvas.paste(key_img, (key_x, 2 * small_h + 2 * gap_h))

    elif num_images == 7:
        if key_frame_index == 2:
            canvas_h = small_h + gap_h + h + gap_h + 2 * small_h + gap_h
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            paste_small_row(canvas, [image_list[0], image_list[1]], [1, 2], 0)

            key_img = add_number_label(image_list[2], 3, True)
            canvas.paste(key_img, (key_x, small_h + gap_h))

            paste_small_row(canvas, [image_list[3], image_list[4]], [4, 5],
                            small_h + gap_h + h + gap_h)
            paste_small_row(canvas, [image_list[5], image_list[6]], [6, 7],
                            small_h + gap_h + h + gap_h + small_h + gap_h)

        else:  # key_frame_index == 4
            canvas_h = 2 * small_h + gap_h + gap_h + h + gap_h + small_h
            canvas = Image.new('RGB', (canvas_w, canvas_h), (240, 240, 240))

            paste_small_row(canvas, [image_list[0], image_list[1]], [1, 2], 0)
            paste_small_row(canvas, [image_list[2], image_list[3]], [3, 4],
                            small_h + gap_h)

            key_img = add_number_label(image_list[4], 5, True)
            canvas.paste(key_img, (key_x, 2 * small_h + 2 * gap_h))

            paste_small_row(canvas, [image_list[5], image_list[6]], [6, 7],
                            2 * small_h + 2 * gap_h + h + gap_h)

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