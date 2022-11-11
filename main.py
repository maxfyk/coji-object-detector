# generate random coji code
# pick random image from backgrounds
# change coji codes size, transform
# place coji code into picked image
# change contrast / brightness, blur
# label (object, x, y, w, h)
# save
import cv2
import numpy as np
import imutils
import random
import torch
import torchvision.transforms as T
from PIL import Image
from processors.generate_coji_codes import generator as coji_gen
from processors.background_images import generator as background_gen

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


def label_img(code_labels, coji_new_xy, coji_pos_x, coji_pos_y, img):
    code_labels_out = []
    coji_point = code_labels[-1][1]
    for label in code_labels:
        points = label[1]
        points_n = []
        for i, point in enumerate(points):
            x = int((point[0] * coji_new_xy[i][0][0]) / coji_point[i][0])
            y = int((point[1] * coji_new_xy[i][0][1]) / coji_point[i][1])
            points_n.append((x, y))
            print(point, coji_new_xy[i][0], coji_point[i], (x, y))

        print('------')
        cv2.rectangle(img, points_n[0], points_n[-1], (254, 254, 254), 3)
        cv2.imshow('i', img)
        cv2.waitKey(0)

    print(code_labels, coji_new_xy, coji_pos_x, coji_pos_y)


def generate_labeled_img():
    code_img, code_labels, code_id = coji_gen.generate_random_code()
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
        img_size * coji_size_prcntg * (code_img.shape[0] / code_img.shape[1]))
    coji_modified = cv2.resize(code_img, (coji_new_size_w, coji_new_size_h), interpolation=cv2.INTER_AREA)
    # change perspective
    color_converted = cv2.cvtColor(coji_modified, cv2.COLOR_BGRA2RGBA)
    coji_pil = Image.fromarray(color_converted)
    # CHANGE IN TRANSFORMS file
    # if torch.rand(1) < self.p:
    #     startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
    #     return F.perspective(img, startpoints, endpoints, self.interpolation, fill), startpoints, endpoints
    perspective_img = perspective_transformer(coji_pil)

    coji_modified = cv2.cvtColor(np.array(perspective_img), cv2.COLOR_BGRA2RGBA)
    contours, hierarchy = cv2.findContours(cv2.cvtColor(np.array(coji_modified), cv2.COLOR_RGB2GRAY),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    coji_new_xy = cv2.approxPolyDP(contours, 0.0035 * cv2.arcLength(contours, True), True)
    #
    # for point in coji_new_xy:
    #     x, y = point[0]
    #     cv2.circle(coji_modified, (x, y), 2, (255, 0, 0), 2)

    # place randomly
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
    label_img(code_labels, coji_new_xy, coji_pos_x, coji_pos_y, coji_modified)

    cv2.imshow('image', coji_modified)
    cv2.imshow('image2', background_img)
    cv2.waitKey(0)
    # label
    # save


if __name__ == '__main__':
    print('loading finished...')
    while True:
        generate_labeled_img()
