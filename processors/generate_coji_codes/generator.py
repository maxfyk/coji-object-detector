# Random char code -> visual code -> return pil/opencv image
import os
from runpy import run_path
import numpy as np
from random import randrange
from PIL import Image
import cv2

STYLE_NAME = 'geom-original'
STYLES_PATH_SHORT = 'statics/styles/{}/'
STYLES_PATH_FULL = STYLES_PATH_SHORT.format(STYLE_NAME)

p = 6364136223846793005
s = 1442695040888963407


def generate_code_id(index: int, style_info: dict, style_module: dict):
    """Generate random code id"""
    n, m = style_info['rows'], style_info['pieces-row']  # dimension of key - n*n
    num_keys = (2 ** m) ** (n * n)  # total number of keys

    sh_idx = (index * p + s) % num_keys  # map to pseudo-random target
    values = [(sh_idx >> (i * m)) & ((1 << m) - 1)
              for i in range(n * n)]  # split into m-bit words
    return ''.join([style_module['keys'][i] for i in values])


def pieces_generator(code_id: str):
    """Yield code_id char by char"""
    for char in code_id:
        yield char


def generate_visual_code(code_id: str, style_module: dict):
    """Visualize string code"""
    style_info = style_module['style-info']
    key_to_name = style_module['key-to-name']

    coji_code = Image.new('RGB', (style_info['size'], style_info['size']), tuple(style_info['background-color']))
    piece_size = int(style_info['size'] / style_info['pieces-row'])

    objects = []
    # add code
    piece_id = pieces_generator(code_id)
    for cur_row in range(style_info['rows']):
        for cur_col in range(style_info['pieces-row']):
            piece_name = key_to_name[next(piece_id)]
            piece = Image.open(
                os.path.join(STYLES_PATH_FULL, 'pieces', f'{piece_name}.png')
            )
            piece = piece.resize((piece_size, piece_size), Image.Resampling.LANCZOS)
            coji_code.paste(piece, (piece_size * cur_col, piece_size * cur_row), piece)
            y = piece_size * cur_col + style_info['template']['template-offset'][0]
            x = piece_size * cur_row + style_info['template']['template-offset'][1]
            objects.append(
                [piece_name, [
                    [x, y],
                    [x, y + piece_size],
                    [x + piece_size, y + piece_size],
                    [x + piece_size, y],
                ]])

            piece.close()

    if style_info['template']['add-template']:
        template = Image.open(
            os.path.join(STYLES_PATH_FULL, 'pieces', 'code-template-v2.jpg')
        )
        template.paste(coji_code, style_info['template']['template-offset'])
        coji_code = template
    x, y = style_info['template']['template-offset']
    objects.append(
        ['coji-code', [
            [x, y],
            [x, y + style_info['size']],
            [x + style_info['size'], y + style_info['size']],
            [x + style_info['size'], y],
        ]])
    objects.append(
        ['coji-frame', [
            [0, 0],
            [0, coji_code.size[1]],
            [coji_code.size[0], coji_code.size[1]],
            [coji_code.size[0], 0],
        ]])
    # with io.BytesIO() as out:
    #     coji_code.save(out, format='JPEG', quality=100, optimize=True)
    #     return encodebytes(out.getvalue()).decode()
    return cv2.cvtColor(np.array(coji_code), cv2.COLOR_RGB2BGR), objects, code_id


def generate_random_code(style_module):
    style_info = style_module['style-info']
    index = randrange(0, 18446744073709551616)
    code_id = generate_code_id(index, style_info, style_module)
    return generate_visual_code(code_id, style_module)


if __name__ == '__main__':
    STYLES_PATH_SHORT = '../../statics/styles/{}/'
    style_module = run_path(os.path.join(STYLES_PATH_FULL, 'properties.py'))['style_module']
    print(generate_random_code(style_module))
