import os
import shutil
from runpy import run_path

STYLE_NAME = 'geom-original'
STYLES_PATH_SHORT = '../../statics/styles/{}/'
STYLES_PATH_FULL = STYLES_PATH_SHORT.format(STYLE_NAME)
style_module = run_path(os.path.join(STYLES_PATH_FULL, 'properties.py'))['style_module']

get_len = lambda key: len(key[0])
name_to_key = style_module['object-detection-model']['name-to-key']
name_to_key_list = list(name_to_key.items())
name_to_key_list.sort(key=get_len, reverse=True)
name_to_key = {e[0]: e[1] for e in name_to_key_list}


def fix_class_names(in_folder_path: str):
    """Adjust class names for yolo"""
    counter_total, counter_transferred = 0, 0

    in_folder_path_os = os.fsencode(in_folder_path)
    for in_file in os.listdir(in_folder_path_os):
        filename = os.fsdecode(in_file)
        if filename.endswith('.txt'):
            print(filename)
            label_path = os.path.join(in_folder_path, filename)
            labels = None
            with open(label_path, 'r') as file:
                labels = file.read()

            for k, v in name_to_key.items():
                labels = labels.replace(k, str(v))

            with open(label_path, 'w') as file:
                file.write(labels)

            counter_total += 1

    print(f'Fixed {counter_total} files')


if __name__ == '__main__':
    fix_class_names('../../data/out/generate_coji_codes/Taipei/clean')
