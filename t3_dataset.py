import os
import numpy as np
import cv2
import random
import math
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from dataset_util import load, show_bbox_on_image


phrase_list = [
    ', content and position of the texts are ',
    ', textual material depicted in the image are ',
    ', texts that says ',
    ', captions shown in the snapshot are ',
    ', with the words of ',
    ', that reads ',
    ', the written materials on the picture: ',
    ', these texts are written on it: ',
    ', captions are ',
    ', content of the text in the graphic is '
]


def insert_spaces(string, nSpace):
    if nSpace == 0:
        return string
    new_string = ""
    for char in string:
        new_string += char + " " * nSpace
    return new_string[:-nSpace]


def draw_glyph(font, text):
    g_size = 50
    W, H = (512, 80)
    new_font = font.font_variant(size=g_size)
    img = Image.new(mode='1', size=(W, H), color=0)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = new_font.getbbox(text)
    text_width = max(right-left, 5)
    text_height = max(bottom - top, 5)
    ratio = min(W*0.9/text_width, H*0.9/text_height)
    new_font = font.font_variant(size=int(g_size*ratio))

    text_width, text_height = new_font.getsize(text)
    offset_x, offset_y = new_font.getoffset(text)
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2 - offset_y//2
    draw.text((x, y), text, font=new_font, fill='white')
    img = np.expand_dims(np.array(img), axis=2).astype(np.float64)
    return img


def draw_glyph2(font, text, polygon, vertAng=10, scale=1, width=512, height=512, add_space=True):
    enlarge_polygon = polygon*scale
    rect = cv2.minAreaRect(enlarge_polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if (abs(angle) % 90 < vertAng or abs(90-abs(angle) % 90) % 90 < vertAng):
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = np.zeros((height*scale, width*scale, 3), np.uint8)
    img = Image.fromarray(img)

    # infer font size
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)
    if text_w <= max(w, h):
        # add space
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i-1)
        font_size = min(w, h)*0.80
    else:
        shrink = 0.75 if vert else 0.85
        font_size = min(w, h) / (text_w/max(w, h)) * shrink
    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right-left
    text_height = bottom - top

    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text((rect[0][0]-text_width//2, rect[0][1]-text_height//2-top), text, font=new_font, fill=(255, 255, 255, 255))
    else:
        x_s = min(box[:, 0]) + _w//2 - text_height//2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))

    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    img = np.expand_dims(np.array(img.convert('1')), axis=2).astype(np.float64)
    return img


def get_caption_pos(ori_caption, pos_idxs, prob=1.0, place_holder='*'):
    idx2pos = {
        0: " top left",
        1: " top",
        2: " top right",
        3: " left",
        4: random.choice([" middle", " center"]),
        5: " right",
        6: " bottom left",
        7: " bottom",
        8: " bottom right"
    }
    new_caption = ori_caption + random.choice(phrase_list)
    pos = ''
    for i in range(len(pos_idxs)):
        if random.random() < prob and pos_idxs[i] > 0:
            pos += place_holder + random.choice([' located', ' placed', ' positioned', '']) + random.choice([' at', ' in', ' on']) + idx2pos[pos_idxs[i]] + ', '
        else:
            pos += place_holder + ' , '
    pos = pos[:-2] + '.'
    new_caption += pos
    return new_caption


def generate_random_rectangles(w, h, box_num):
    rectangles = []
    for i in range(box_num):
        x = random.randint(0, w)
        y = random.randint(0, h)
        w = random.randint(16, 256)
        h = random.randint(16, 96)
        angle = random.randint(-45, 45)
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x + w, y + h)
        p4 = (x, y + h)
        center = ((x + x + w) / 2, (y + y + h) / 2)
        p1 = rotate_point(p1, center, angle)
        p2 = rotate_point(p2, center, angle)
        p3 = rotate_point(p3, center, angle)
        p4 = rotate_point(p4, center, angle)
        rectangles.append((p1, p2, p3, p4))
    return rectangles


def rotate_point(point, center, angle):
    # rotation
    angle = math.radians(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    x1 += center[0]
    y1 += center[1]
    return int(x1), int(y1)


class T3DataSet(Dataset):
    def __init__(
            self,
            json_path,
            max_lines=5,
            max_chars=20,
            place_holder='*',
            font_path='./font/Arial_Unicode.ttf',
            caption_pos_prob=1.0,
            mask_pos_prob=1.0,
            mask_img_prob=0.5,
            for_show=False,
            using_dlc=False,
            glyph_scale=1,
            percent=1.0,
            debug=False,
            wm_thresh=1.0,
            ):
        assert isinstance(json_path, (str, list))
        if isinstance(json_path, str):
            json_path = [json_path]
        data_list = []
        self.using_dlc = using_dlc
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.place_holder = place_holder
        self.font = ImageFont.truetype(font_path, size=60)
        self.caption_pos_porb = caption_pos_prob
        self.mask_pos_prob = mask_pos_prob
        self.mask_img_prob = mask_img_prob
        self.for_show = for_show
        self.glyph_scale = glyph_scale
        self.wm_thresh = wm_thresh
        for jp in json_path:
            data_list += self.load_data(jp, percent)
        self.data_list = data_list
        print(f'All dataset loaded, imgs={len(self.data_list)}')
        self.debug = debug
        if self.debug:
            self.tmp_items = [i for i in range(100)]

    def load_data(self, json_path, percent):
        content = load(json_path)
        d = []
        count = 0
        wm_skip = 0
        max_img = len(content['data_list']) * percent
        for gt in content['data_list']:
            if len(d) > max_img:
                break
            if 'wm_score' in gt and gt['wm_score'] > self.wm_thresh:  # wm_score > thresh will be skiped as an img with watermark
                wm_skip += 1
                continue
            data_root = content['data_root']
            if self.using_dlc:
                data_root = data_root.replace('/data/vdb', '/mnt/data', 1)
            img_path = os.path.join(data_root, gt['img_name'])
            info = {}
            info['img_path'] = img_path
            info['caption'] = gt['caption'] if 'caption' in gt else ''
            if self.place_holder in info['caption']:
                count += 1
                info['caption'] = info['caption'].replace(self.place_holder, " ")
            if 'annotations' in gt:
                polygons = []
                invalid_polygons = []
                texts = []
                languages = []
                pos = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0:
                        continue
                    if 'valid' in annotation and annotation['valid'] is False:
                        invalid_polygons.append(annotation['polygon'])
                        continue
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    languages.append(annotation['language'])
                    if 'pos' in annotation:
                        pos.append(annotation['pos'])
                info['polygons'] = [np.array(i) for i in polygons]
                info['invalid_polygons'] = [np.array(i) for i in invalid_polygons]
                info['texts'] = texts
                info['language'] = languages
                info['pos'] = pos
            d.append(info)
        print(f'{json_path} loaded, imgs={len(d)}, wm_skip={wm_skip}')
        if count > 0:
            print(f"Found {count} image's caption contain placeholder: {self.place_holder}, change to ' '...")
        return d

    def __getitem__(self, item):
        item_dict = {}
        if self.debug:  # sample fixed items
            item = self.tmp_items.pop()
            print(f'item = {item}')
        cur_item = self.data_list[item]
        # img
        target = np.array(Image.open(cur_item['img_path']).convert('RGB'))
        if target.shape[0] != 512 or target.shape[1] != 512:
            target = cv2.resize(target, (512, 512))
        target = (target.astype(np.float32) / 127.5) - 1.0
        item_dict['img'] = target
        # caption
        item_dict['caption'] = cur_item['caption']
        item_dict['glyphs'] = []
        item_dict['gly_line'] = []
        item_dict['positions'] = []
        item_dict['texts'] = []
        item_dict['language'] = []
        item_dict['inv_mask'] = []
        texts = cur_item.get('texts', [])
        if len(texts) > 0:
            idxs = [i for i in range(len(texts))]
            if len(texts) > self.max_lines:
                sel_idxs = random.sample(idxs, self.max_lines)
                unsel_idxs = [i for i in idxs if i not in sel_idxs]
            else:
                sel_idxs = idxs
                unsel_idxs = []
            if len(cur_item['pos']) > 0:
                pos_idxs = [cur_item['pos'][i] for i in sel_idxs]
            else:
                pos_idxs = [-1 for i in sel_idxs]
            item_dict['caption'] = get_caption_pos(item_dict['caption'], pos_idxs, self.caption_pos_porb, self.place_holder)
            item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
            item_dict['texts'] = [cur_item['texts'][i][:self.max_chars] for i in sel_idxs]
            item_dict['language'] = [cur_item['language'][i] for i in sel_idxs]
            # glyphs
            for idx, text in enumerate(item_dict['texts']):
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(self.font, text, item_dict['polygons'][idx], scale=self.glyph_scale)
                item_dict['glyphs'] += [glyphs]
                item_dict['gly_line'] += [gly_line]
            # mask_pos
            for polygon in item_dict['polygons']:
                item_dict['positions'] += [self.draw_pos(polygon, self.mask_pos_prob)]
        # inv_mask
        invalid_polygons = cur_item['invalid_polygons'] if 'invalid_polygons' in cur_item else []
        if len(texts) > 0:
            invalid_polygons += [cur_item['polygons'][i] for i in unsel_idxs]
        item_dict['inv_mask'] = self.draw_inv_mask(invalid_polygons)
        item_dict['hint'] = self.get_hint(item_dict['positions'])
        if random.random() < self.mask_img_prob:
            # randomly generate 0~3 masks
            box_num = random.randint(0, 3)
            boxes = generate_random_rectangles(512, 512, box_num)
            boxes = np.array(boxes)
            pos_list = item_dict['positions'].copy()
            for i in range(box_num):
                pos_list += [self.draw_pos(boxes[i], self.mask_pos_prob)]
            mask = self.get_hint(pos_list)
            masked_img = target*(1-mask)
        else:
            masked_img = np.zeros_like(target)
        item_dict['masked_img'] = masked_img

        if self.for_show:
            item_dict['img_name'] = os.path.split(cur_item['img_path'])[-1]
            return item_dict
        if len(texts) > 0:
            del item_dict['polygons']
        # padding
        n_lines = min(len(texts), self.max_lines)
        item_dict['n_lines'] = n_lines
        n_pad = self.max_lines - n_lines
        if n_pad > 0:
            item_dict['glyphs'] += [np.zeros((512*self.glyph_scale, 512*self.glyph_scale, 1))] * n_pad
            item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
            item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
            item_dict['texts'] += [' '] * n_pad
            item_dict['language'] += [' '] * n_pad

        return item_dict

    def __len__(self):
        return len(self.data_list)

    def draw_inv_mask(self, polygons):
        img = np.zeros((512, 512))
        for p in polygons:
            pts = p.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
        img = img[..., None]
        return img/255.

    def draw_pos(self, ploygon, prob=1.0):
        img = np.zeros((512, 512))
        rect = cv2.minAreaRect(ploygon)
        w, h = rect[1]
        small = False
        if w < 20 or h < 20:
            small = True
        if random.random() < prob:
            pts = ploygon.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
            # 10% dilate / 10% erode / 5% dilatex2  5% erodex2
            random_value = random.random()
            kernel = np.ones((3, 3), dtype=np.uint8)
            if random_value < 0.7:
                pass
            elif random_value < 0.8:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.9 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.95:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=2)
            elif random_value < 1.0 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=2)
        img = img[..., None]
        return img/255.

    def get_hint(self, positions):
        if len(positions) == 0:
            return np.zeros((512, 512, 1))
        return np.sum(positions, axis=0).clip(0, 1)


if __name__ == '__main__':
    '''
    Run this script to show details of your dataset, such as ocr annotations, glyphs, prompts, etc.
    '''
    from tqdm import tqdm
    from matplotlib import pyplot as plt
    import shutil

    show_imgs_dir = 'show_results'
    show_count = 50
    if os.path.exists(show_imgs_dir):
        shutil.rmtree(show_imgs_dir)
    os.makedirs(show_imgs_dir)
    plt.rcParams['axes.unicode_minus'] = False
    json_paths = [
        '/path/of/your/dataset/data1.json',
        '/path/of/your/dataset/data2.json',
        # ...
    ]

    dataset = T3DataSet(json_paths, for_show=True, max_lines=20, glyph_scale=2, mask_img_prob=1.0, caption_pos_prob=0.0)
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    pbar = tqdm(total=show_count)
    for i, data in enumerate(train_loader):
        if i == show_count:
            break
        img = ((data['img'][0].numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        masked_img = ((data['masked_img'][0].numpy() + 1.0) / 2.0 * 255)[..., ::-1].astype(np.uint8)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_masked.jpg'), masked_img)
        if 'texts' in data and len(data['texts']) > 0:
            texts = [x[0] for x in data['texts']]
            img = show_bbox_on_image(Image.fromarray(img), data['polygons'], texts)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}.jpg'),  np.array(img)[..., ::-1])
        with open(os.path.join(show_imgs_dir, f'plots_{i}.txt'), 'w') as fin:
            fin.writelines([data['caption'][0]])
        all_glyphs = []
        for k, glyphs in enumerate(data['glyphs']):
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_glyph_{k}.jpg'), glyphs[0].numpy().astype(np.int32)*255)
            all_glyphs += [glyphs[0].numpy().astype(np.int32)*255]
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_allglyphs.jpg'), np.sum(all_glyphs, axis=0))
        for k, gly_line in enumerate(data['gly_line']):
            cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_gly_line_{k}.jpg'), gly_line[0].numpy().astype(np.int32)*255)
        for k, position in enumerate(data['positions']):
            if position is not None:
                cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_pos_{k}.jpg'), position[0].numpy().astype(np.int32)*255)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_hint.jpg'), data['hint'][0].numpy().astype(np.int32)*255)
        cv2.imwrite(os.path.join(show_imgs_dir, f'plots_{i}_inv_mask.jpg'), np.array(img)[..., ::-1]*(1-data['inv_mask'][0].numpy().astype(np.int32)))
        pbar.update(1)
    pbar.close()
