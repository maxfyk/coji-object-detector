import os
import shutil
from runpy import run_path
from multiprocessing import Process
from pathlib import Path
from sklearn.model_selection import train_test_split

STYLE_NAME = 'geom-original'
STYLES_PATH_SHORT = '../../statics/styles/{}/'
STYLES_PATH_FULL = STYLES_PATH_SHORT.format(STYLE_NAME)


def move_files_to_folder(file_names, in_folder, out_folder):
    for f_path in file_names:
        f_name = f_path.replace(in_folder, '').replace('\\', '').replace('/', '')
        shutil.copyfile(f_path, os.path.join(out_folder, f_name))


def partition_dataset(in_folder_path: str, out_folder_path: str):
    in_folder_path_os = os.fsencode(in_folder_path)
    # Read images and labels
    images = [os.path.join(in_folder_path, 'images', x) for x in os.listdir(os.path.join(in_folder_path, 'images'))]
    labels = [os.path.join(in_folder_path, 'labels', x) for x in
                   os.listdir(os.path.join(in_folder_path, 'labels'))
                   if x[-3:] == "txt"]

    images.sort()
    labels.sort()

    # Split the dataset into train-valid-test splits
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2,
                                                                                    random_state=1)
    val_images, test_images, val_labels, test_labels = train_test_split(val_images, val_labels,
                                                                                  test_size=0.5, random_state=1)
    tasks = []
    tasks.append(Process(target=move_files_to_folder, args=(train_images, os.path.join(in_folder_path, 'images'),
                                                             os.path.join(out_folder_path, 'images/train'))))
    tasks.append(Process(target=move_files_to_folder, args=(val_images, os.path.join(in_folder_path, 'images'),
                                                             os.path.join(out_folder_path, 'images/validate/'))))
    tasks.append(Process(target=move_files_to_folder, args=(test_images, os.path.join(in_folder_path, 'images'),
                                                             os.path.join(out_folder_path, 'images/test/'))))
    tasks.append(Process(target=move_files_to_folder,
                 args=(train_labels, os.path.join(in_folder_path, 'labels'),
                       os.path.join(out_folder_path, 'labels/train/'))))
    tasks.append(Process(target=move_files_to_folder,
                 args=(val_labels, os.path.join(in_folder_path, 'labels'),
                       os.path.join(out_folder_path, 'labels/validate/'))))
    tasks.append(Process(target=move_files_to_folder,
                 args=(test_labels, os.path.join(in_folder_path, 'labels'),
                       os.path.join(out_folder_path, 'labels/test/'))))
    [f.start() for f in tasks]
    [f.join() for f in tasks]


if __name__ == '__main__':
    in_path = '../../data/out/generate_coji_codes/Taipei/splitted'
    out_path = '../../data/out/generate_coji_codes/Taipei/partitioned'
    Path(os.path.join(out_path, 'images')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'labels')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'images', 'train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'images', 'test')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'images', 'validate')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'labels', 'train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'labels', 'test')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_path, 'labels', 'validate')).mkdir(parents=True, exist_ok=True)
    partition_dataset(in_path, out_path)
