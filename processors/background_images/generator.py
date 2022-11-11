import os
import random
import cv2


def get_random_background_image(backgrounds_path):
    background_name = random.choice(os.listdir(backgrounds_path))
    return cv2.imread(os.path.join(backgrounds_path, background_name))
