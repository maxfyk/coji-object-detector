# generate random coji code
# pick random image from backgrounds
# change coji codes size, transform
# place coji code into picked image
# change contrast / brightness, blur
# label (object, x, y, w, h)
# save
import os
import random
import math
from runpy import run_path
import cv2
import numpy as np
import torch
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import multiprocessing
from processors.background_images import generator as background_gen
from processors.generate_coji_codes import generator as coji_gen

BACKGROUNDS_PATH = 'data/out/background_images/Taipei'
torch.manual_seed(0)
jitter = T.ColorJitter(brightness=.6, contrast=.5, saturation=.25, hue=.05)
blurrer = T.GaussianBlur(kernel_size=(13, 17), sigma=(0.1, 2))
perspective_transformer = T.RandomPerspective(distortion_scale=0.7, p=0.8)

STYLE_NAME = 'geom-original'
STYLES_PATH_SHORT = 'statics/styles/{}/'
STYLES_PATH_FULL = STYLES_PATH_SHORT.format(STYLE_NAME)
style_module = run_path(os.path.join(STYLES_PATH_FULL, 'properties.py'))['style_module']
style_info = style_module['style-info']
name_to_key = style_module['object-detection-model']['name-to-key']


def label_img(code_labels, matrix, coji_pos_y, coji_pos_x, back_img_w, back_img_h):
    """"""
    img_center = (back_img_w / 2), (back_img_h / 2)
    code_labels_out = []
    for label in code_labels:
        points = label[1]
        points_n = []
        for i, p in enumerate(points):
            px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
                (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
            py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
                (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
            px, py = float(px) + coji_pos_x, float(py) + coji_pos_y
            points_n.append((px, py))
        code_labels_out.append([label[0], points_n])
    return code_labels_out


def resize_original_labels(coji_new_size_w, coji_new_size_h, coji_size_h, coji_size_w, code_labels):
    """Resize labels based on new coji code size"""
    code_labels_out = []
    for label in code_labels:
        points = label[1]
        points_n = []
        for i, point in enumerate(points):
            points_n.append(
                [float((point[0] * coji_new_size_w) / coji_size_w), float((point[1] * coji_new_size_h) / coji_size_h)]
            )
        code_labels_out.append([
            label[0], points_n
        ])
    return code_labels_out


def generate_labeled_img(i):
    global out_path, out
    # get random coji code
    code_img, code_labels, code_id = coji_gen.generate_random_code(style_module)
    code_img = cv2.cvtColor(code_img, cv2.COLOR_RGB2RGBA)

    # get random background image
    background_img = background_gen.get_random_background_image(BACKGROUNDS_PATH)
    background_img_h, background_img_w = background_img.shape[:2]

    background_img = cv2.resize(background_img, (int(background_img_w / 3), int(background_img_h / 3)),
                                interpolation=cv2.INTER_AREA)
    background_img_h, background_img_w = background_img.shape[:2]

    # get min background_img side
    background_img = cv2.cvtColor(background_img, cv2.COLOR_RGB2RGBA)
    img_size = min(background_img.shape[:2])
    coji_size_prcntg = random.uniform(0.16, 0.88)
    # make coji N - N +100 % of it
    coji_new_size_h, coji_new_size_w = int(img_size * coji_size_prcntg), int(
        img_size * coji_size_prcntg * (code_img.shape[1] / code_img.shape[0]))
    coji_modified = cv2.resize(code_img, (coji_new_size_w, coji_new_size_h), interpolation=cv2.INTER_AREA)
    # cv2.imwrite('code_img.png', coji_modified)

    # adjust labels for new size

    code_labels = resize_original_labels(coji_new_size_w, coji_new_size_h, *code_img.shape[:2], code_labels)

    # change perspective
    color_converted = cv2.cvtColor(coji_modified, cv2.COLOR_BGRA2RGBA)
    coji_pil = Image.fromarray(color_converted)
    # get random transform
    pts1, pts2 = perspective_transformer.get_params(*coji_pil.size, 0.8)
    pts1, pts2 = np.float32(pts1), np.float32(pts2)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # transform
    coji_modified = cv2.warpPerspective(coji_modified, matrix, coji_pil.size)
    coji_modified = cv2.cvtColor(coji_modified, cv2.COLOR_RGB2RGBA)

    # paste coji code into a random place in an image
    coji_modified_h, coji_modified_w = coji_modified.shape[:2]
    coji_pos_x = random.randint(0, background_img_w - coji_modified_w)
    coji_pos_y = random.randint(0, background_img_h - coji_modified_h)

    alpha_s = coji_modified[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        background_img[coji_pos_y:coji_pos_y + coji_modified_h, coji_pos_x:coji_pos_x + coji_modified_w, c] = (
                alpha_s * coji_modified[:, :, c] +
                alpha_l * background_img[coji_pos_y:coji_pos_y + coji_modified_h,
                          coji_pos_x:coji_pos_x + coji_modified_w, c])

    # blur, change colors, etc.
    background_img_pil = cv2.cvtColor(background_img, cv2.COLOR_BGRA2RGBA)
    background_img_pil = Image.fromarray(background_img_pil)

    jitted_img = jitter(background_img_pil)
    jitted_img = blurrer(jitted_img)
    background_img = cv2.cvtColor(np.array(jitted_img), cv2.COLOR_RGBA2BGRA)
    code_labels = label_img(code_labels, matrix, coji_pos_y, coji_pos_x, background_img_w, background_img_h)

    category = 'TRAIN'
    rand_cat = random.randint(1, 10)
    if rand_cat in (8, 9):
        category = 'TEST'
    elif rand_cat == 10:
        category = 'VALIDATION'
    for label in code_labels:
        name, points = label
        if name != 'coji-code':
            continue
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        max_x, max_y = max(xs), max(ys)
        min_x, min_y = min(xs), min(ys)
        w = math.hypot(max_x - min_x, min_y - min_y)
        h = math.hypot(min_x - min_x, max_y - min_y)
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        img_path = os.path.join('images', f'{code_id}.jpg')
        out.write(
            # f'{category},{img_path},{name},{(min_x / background_img_w)},{(min_y / background_img_h)},,,{(max_x / background_img_w)},{(max_y / background_img_h)},,\n')
            f'{category},{img_path},{name},{(xs[0] / background_img_w)},{(ys[0] / background_img_w)},,,{(xs[2] / background_img_w)},{(ys[2] / background_img_w)},,\n')

    cv2.imwrite(os.path.join(out_path, 'images', f'{code_id}.jpg'), background_img)
    # cv2.imshow('image', coji_modified)
    # cv2.imshow('image2', background_img)
    # cv2.waitKey(0)
    # label
    # save


if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool

    out_path = 'data/out/generate_coji_codes/'
    Path(os.path.join(out_path)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'images')).mkdir(parents=True, exist_ok=True)
    out = open(os.path.join(out_path, f'dataset.csv'), 'w+')

    print('loading finished...')
    pool = ThreadPool()
    for i in range(10):
        print('started', i)
        res = pool.map(generate_labeled_img, range(500))  # 50000
    out.close()