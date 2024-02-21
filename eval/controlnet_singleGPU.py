'''
Part of the implementation is borrowed and modified from ControlNet, publicly available at https://github.com/lllyasviel/ControlNet/blob/main/gradio_canny2image.py
'''
from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # specify the inference settings
    parser.add_argument(
        "--model_path",
        type=str,
        default='/home/yuxiang.tyx/projects/AnyText/models/control_sd15_canny.pth',
        help='path of model'
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--a_prompt",
        type=str,
        default='best quality, extremely detailed',
        help="additional prompt"
    )
    parser.add_argument(
        "--n_prompt",
        type=str,
        default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark',
        help="negative prompt"
    )
    parser.add_argument(
        "--image_resolution",
        type=int,
        default=512,
        help="image resolution",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1,
        help="control strength",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="classifier-free guidance scale",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="ddim steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="seed",
    )
    parser.add_argument(
        "--guess_mode",
        action="store_true",
        help="whether use guess mode",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0,
        help="eta",
    )
    parser.add_argument(
        "--low_threshold",
        type=int,
        default=100,
        help="low threshold",
    )
    parser.add_argument(
        "--high_threshold",
        type=int,
        default=200,
        help="high threshold",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./controlnet_laion_generated/',
        help="output path"
    )
    parser.add_argument(
        "--glyph_dir",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/laion_word/glyph_laion',
        help="path of glyph images from anytext evaluation dataset"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/laion_word/test1k.json',
        help="json path for evaluation dataset"
    )

    args = parser.parse_args()
    return args


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, scale, seed, eta, low_threshold, high_threshold, model, ddim_sampler):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples, 
                                                     shape, cond, mask=None,
                                                     x0=None, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

    return results


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


def load_data(input_path):
    content = load_json(input_path)
    d = []
    count = 0
    for gt in content['data_list']:
        info = {}
        info['img_name'] = gt['img_name']
        info['caption'] = gt['caption']
        if PLACE_HOLDER in info['caption']:
            count += 1
            info['caption'] = info['caption'].replace(PLACE_HOLDER, " ")
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0:
                    continue
                if annotation['valid'] is False:
                    continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                pos.append(annotation['pos'])
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    print(f'{input_path} loaded, imgs={len(d)}')
    if count > 0:
        print(f"Found {count} image's caption contain placeholder: {PLACE_HOLDER}, change to ' '...")
    return d


def get_item(data_list, item):
    item_dict = {}
    cur_item = data_list[item]
    item_dict['img_name'] = cur_item['img_name']
    item_dict['caption'] = cur_item['caption']
    return item_dict


if __name__ == "__main__":
    args = parse_args()

    apply_canny = CannyDetector()
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(args.model_path, location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    if os.path.exists(args.output_dir) is not True:
        os.makedirs(args.output_dir)

    PLACE_HOLDER = '*'
    data_list = load_data(args.json_path)

    for i in tqdm(range(len(data_list)), desc='generator'):
        item_dict = get_item(data_list, i)
        p = item_dict['img_name']
        img_name = item_dict['img_name'].split('.')[0] + '_3.jpg'
        if os.path.exists(os.path.join(args.output_dir, img_name)):
            continue
        input_image_path = os.path.join(args.glyph_dir, p)
        prompt = item_dict['caption']

        img = Image.open(input_image_path)
        input_image = np.array(img)
        results = process(input_image, prompt, args.a_prompt, args.n_prompt, args.num_samples, args.image_resolution, args.ddim_steps, args.scale, args.seed, args.eta, args.low_threshold, args.high_threshold, model, ddim_sampler)
        for idx, img in enumerate(results):
            img_name = item_dict['img_name'].split('.')[0]+f'_{idx}' + '.jpg'
            cv2.imwrite(os.path.join(args.output_dir, img_name), img[..., ::-1])
