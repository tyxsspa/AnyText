"""
Part of the implementation is borrowed and modified from GlyphControl, publicly available at https://github.com/AIGText/GlyphControl-release/blob/main/inference.py
"""
import torch
from PIL import Image
from cldm.hack import disable_verbosity, enable_sliced_attention
from scripts.rendertext_tool import load_model_from_config
from omegaconf import OmegaConf
import argparse
import os
import json
import random
import einops
from tqdm import tqdm
import numpy as np
import cv2
from pytorch_lightning import seed_everything
from contextlib import nullcontext
from cldm.ddim_hacked import DDIMSampler
from torchvision.transforms import ToTensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/config.yaml",
        help="path to model config",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='checkpoints/laion10M_epoch_6_model_ema_only.ckpt',
        help='path to checkpoint of model'
    )
    parser.add_argument(
        "--save_memory",
        action="store_true",
        default=False,
        help="whether to save memory by transferring some unused parts of models to the cpu device during inference",
    )
    # specify the inference settings
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
        "--output_dir",
        type=str,
        default='./glyphcontrol_laion_generated',
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


class Render_Text:
    def __init__(self,
                 model,
                 precision_scope=nullcontext,
                 transform=ToTensor(),
                 save_memory=False,
                 ):
        self.model = model
        self.precision_scope = precision_scope
        self.transform = transform
        self.ddim_sampler = DDIMSampler(model)
        self.save_memory = save_memory

    def process_multi(self,
                      shared_prompt,
                      glyph_img_path,
                      shared_num_samples, shared_image_resolution,
                      shared_ddim_steps, shared_guess_mode,
                      shared_strength, shared_scale, shared_seed,
                      shared_eta, shared_a_prompt, shared_n_prompt,
                      only_show_rendered_image=False,
                      font_name="calibri"
                      ):
        if shared_seed == -1:
            shared_seed = random.randint(0, 65535)
        seed_everything(shared_seed)
        with torch.no_grad(), self.precision_scope("cuda"), self.model.ema_scope("Sampling on Benchmark Prompts"):
            whiteboard_img = Image.open(glyph_img_path).convert("RGB")
            control = self.transform(whiteboard_img.copy())
            if torch.cuda.is_available():
                control = control.cuda()
            control = torch.stack([control for _ in range(shared_num_samples)], dim=0)
            control = control.clone()
            control = [control]

            H, W = shared_image_resolution, shared_image_resolution
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", False)
                self.model.low_vram_shift(is_diffusing=False)

            print("control is None: {}".format(control is None))
            if shared_prompt.endswith("."):
                if shared_a_prompt == "":
                    c_prompt = shared_prompt
                else:
                    c_prompt = shared_prompt + " " + shared_a_prompt
            elif shared_prompt.endswith(","):
                if shared_a_prompt == "":
                    c_prompt = shared_prompt[:-1] + "."
                else:
                    c_prompt = shared_prompt + " " + shared_a_prompt
            else:
                if shared_a_prompt == "":
                    c_prompt = shared_prompt + "."
                else:
                    c_prompt = shared_prompt + ", " + shared_a_prompt
            cond_c_cross = self.model.get_learned_conditioning([c_prompt] * shared_num_samples)
            print("prompt:", c_prompt)
            un_cond_cross = self.model.get_learned_conditioning([shared_n_prompt] * shared_num_samples)
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", True)
                self.model.low_vram_shift(is_diffusing=True)

            cond = {"c_concat": control, "c_crossattn": [cond_c_cross] if not isinstance(cond_c_cross, list) else cond_c_cross}
            un_cond = {"c_concat": None if shared_guess_mode else control, "c_crossattn": [un_cond_cross] if not isinstance(un_cond_cross, list) else un_cond_cross}
            shape = (4, H // 8, W // 8)

            if not self.model.learnable_conscale:
                self.model.control_scales = [shared_strength * (0.825 ** float(12 - i)) for i in range(13)] if shared_guess_mode else ([shared_strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            else:
                print("learned control scale: {}".format(str(self.model.control_scales)))
            samples, intermediates = self.ddim_sampler.sample(shared_ddim_steps, shared_num_samples,
                                                              shape, cond, verbose=False, eta=shared_eta,
                                                              unconditional_guidance_scale=shared_scale,
                                                              unconditional_conditioning=un_cond)
            if torch.cuda.is_available() and self.save_memory:
                print("low_vram_shift: is_diffusing", False)
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(shared_num_samples)]
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
    disable_verbosity()
    if args.save_memory:
        enable_sliced_attention()
    cfg = OmegaConf.load(args.cfg)
    model = load_model_from_config(cfg, args.model_path, verbose=True)
    render_tool = Render_Text(model, save_memory=args.save_memory)
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
        results = render_tool.process_multi(prompt, input_image_path, args.num_samples, args.image_resolution, args.ddim_steps, args.guess_mode, args.strength, args.scale, args.seed, args.eta, args.a_prompt, args.n_prompt)
        for idx, img in enumerate(results):
            img_name = item_dict['img_name'].split('.')[0]+f'_{idx}' + '.jpg'
            cv2.imwrite(os.path.join(args.output_dir, img_name), img[..., ::-1])
