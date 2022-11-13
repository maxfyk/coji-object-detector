import os
import shutil
from runpy import run_path
from pathlib import Path

STYLE_NAME = 'geom-original'
STYLES_PATH_SHORT = '../../statics/styles/{}/'
STYLES_PATH_FULL = STYLES_PATH_SHORT.format(STYLE_NAME)


def split_imgs_and_txts(in_file):
    filename = os.fsdecode(in_file)
    if filename == 'classes.txt':
        shutil.copyfile(os.path.join(in_folder_path, filename), os.path.join(out_folder_path, filename))
    elif filename.endswith('.txt'):
        shutil.copyfile(os.path.join(in_folder_path, filename), os.path.join(out_folder_path, 'labels', filename))
    elif filename.endswith('.jpg'):
        shutil.copyfile(os.path.join(in_folder_path, filename), os.path.join(out_folder_path, 'images', filename))
    else:
        print(f'Wrong file: {filename}')


if __name__ == '__main__':
    from multiprocessing.pool import ThreadPool

    in_folder_path = '../../data/out/generate_coji_codes/Taipei/clean'
    out_folder_path = '../../data/out/generate_coji_codes/Taipei/splitted'
    in_folder_path_os = os.fsencode(in_folder_path)
    Path(os.path.join(out_folder_path, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_folder_path, 'labels')).mkdir(parents=True, exist_ok=True)
    pool = ThreadPool(processes=32)
    res = pool.map(split_imgs_and_txts, iter(os.listdir(in_folder_path_os)))  # 50000
