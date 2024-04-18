import ujson
import json
import pathlib

__all__ = ['load', 'save', 'show_bbox_on_image']


def load(file_path: str):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': load_txt, '.json': load_json, '.list': load_txt}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](file_path)


def load_txt(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = [x.strip().strip('\ufeff').strip('\xef\xbb\xbf') for x in f.readlines()]
    return content


def load_json(file_path: str):
    with open(file_path, 'rb') as f:
        content = f.read()
    return ujson.loads(content)


def save(data, file_path):
    file_path = pathlib.Path(file_path)
    func_dict = {'.txt': save_txt, '.json': save_json}
    assert file_path.suffix in func_dict
    return func_dict[file_path.suffix](data, file_path)


def save_txt(data, file_path):
    if not isinstance(data, list):
        data = [data]
    with open(file_path, mode='w', encoding='utf8') as f:
        f.write('\n'.join(data))


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def show_bbox_on_image(image, polygons=None, txt=None, color=None, font_path='./font/Arial_Unicode.ttf'):
    from PIL import ImageDraw, ImageFont
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    if len(txt) == 0:
        txt = None
    if color is None:
        color = (255, 0, 0)
    if txt is not None:
        font = ImageFont.truetype(font_path, 20)
    for i, box in enumerate(polygons):
        box = box[0]
        if txt is not None:
            draw.text((int(box[0][0]) + 20, int(box[0][1]) - 20), str(txt[i]), fill='red', font=font)
        for j in range(len(box) - 1):
            draw.line((box[j][0], box[j][1], box[j + 1][0], box[j + 1][1]), fill=color, width=2)
        draw.line((box[-1][0], box[-1][1], box[0][0], box[0][1]), fill=color, width=2)
    return image


def show_glyphs(glyphs, name):
    import numpy as np
    import cv2
    size = 64
    gap = 5
    n_char = 20
    canvas = np.ones((size, size*n_char + gap*(n_char-1), 1))*0.5
    x = 0
    for i in range(glyphs.shape[-1]):
        canvas[:, x:x + size, :] = glyphs[..., i:i+1]
        x += size+gap
    cv2.imwrite(name, canvas*255)
