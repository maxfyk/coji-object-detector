import os
import shutil
from PIL import Image, ExifTags


def filter_background_images(in_folder_path: str, out_folder_path: str):
    """Copy background images, skipping videos, landscapes and duplicates from in folder"""
    counter_total, counter_transferred = 0, 0

    in_folder_path_os = os.fsencode(in_folder_path)
    for in_file in os.listdir(in_folder_path_os):
        filename = os.fsdecode(in_file)
        if filename.endswith('.jpg'):
            img_path = os.path.join(in_folder_path, filename)
            img = Image.open(img_path)
            if img.size[0] > img.size[1]:
                # if landscape
                continue
            shutil.copyfile(img_path, os.path.join(out_folder_path, filename))
            counter_transferred += 1
        counter_total += 1

    print(f'Transferred {counter_transferred}/{counter_total} files')


if __name__ == '__main__':
    filter_background_images('../../data/in/background_images/Taipei', '../../data/out/background_images/Taipei')
