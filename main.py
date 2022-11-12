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
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import multiprocessing
from processors.background_images import generator as background_gen
from processors.generate_coji_codes import generator as coji_gen

BACKGROUNDS_PATH = 'data/out/background_images/Taipei'
torch.manual_seed(0)
jitter = T.ColorJitter(brightness=.6, contrast=.5, saturation=.25, hue=.05)
blurrer = T.GaussianBlur(kernel_size=(13, 17), sigma=(0.1, 2))
perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)


def rotate_img(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def label_img(code_labels, matrix, coji_pos_y, coji_pos_x, final_img):
    """"""
    code_labels_out = []
    img = final_img.copy()
    for label in code_labels:
        points = label[1]
        points_n = []
        for i, p in enumerate(points):
            px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
                (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
            py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
                (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
            points_n.append((int(px) + coji_pos_x, int(py) + coji_pos_y))
            img = cv2.circle(img, points_n[-1], radius=1, color=(0, 0, 255), thickness=4)
        code_labels_out.append([label[0], points_n])
    return code_labels_out, img


def resize_original_labels(coji_new_size_w, coji_new_size_h, coji_size_h, coji_size_w, code_labels):
    """Resize labels based on new coji code size"""
    code_labels_out = []
    for label in code_labels:
        points = label[1]
        points_n = []
        for i, point in enumerate(points):
            points_n.append(
                [round((point[0] * coji_new_size_w) / coji_size_w), round((point[1] * coji_new_size_h) / coji_size_h)]
            )
        code_labels_out.append([
            label[0], points_n
        ])
    return code_labels_out


def generate_labeled_img(i):
    global out_path
    # get random coji code
    code_img, code_labels, code_id = coji_gen.generate_random_code()
    code_img = cv2.cvtColor(code_img, cv2.COLOR_RGB2RGBA)

    # get random background image
    background_img = background_gen.get_random_background_image(BACKGROUNDS_PATH)
    background_img_h, background_img_w = background_img.shape[:2]

    background_img = cv2.resize(background_img, (int(background_img_w / 6), int(background_img_h / 6)),
                                interpolation=cv2.INTER_AREA)
    background_img_h, background_img_w = background_img.shape[:2]

    # get min background_img side
    background_img = cv2.cvtColor(background_img, cv2.COLOR_RGB2RGBA)
    img_size = min(background_img.shape[:2])
    coji_size_prcntg = random.uniform(0.26, 0.88)
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
    pts1, pts2 = perspective_transformer.get_params(*coji_pil.size, 0.5)
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
    code_labels, labeled_img = label_img(code_labels, matrix, coji_pos_y, coji_pos_x, background_img)
    with open(os.path.join(out_path, 'clean', f'{code_id}.txt'), 'w+') as out:
        for label in code_labels:
            name, points = label
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            max_x, max_y = max(xs), max(ys)
            min_x, min_y = min(xs), min(ys)
            w = math.hypot(max_x - min_x, min_y - min_y)
            h = math.hypot(min_x - min_x, max_y - min_y)
            center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
            out.write(f'{name} {int(w)} {int(h)} {int(center[0])} {int(center[1])}\n')

    cv2.imwrite(os.path.join(out_path, 'clean', f'{code_id}.jpg'), background_img)
    cv2.imwrite(os.path.join(out_path, 'labeled', f'{code_id}.jpg'), labeled_img)
    print(code_id, i)
    # cv2.imshow('image', coji_modified)
    # cv2.imshow('image2', background_img)
    # cv2.waitKey(0)
    # label
    # save


out_path = 'data/out/generate_coji_codes/Taipei'

if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool

    print('loading finished...')
    pool = ThreadPool()
    res = pool.map(generate_labeled_img, range(50000))  # 50000
